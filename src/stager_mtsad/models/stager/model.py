from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_backbone import TemporalDAG
from .conditioning import (
    GlobalEmbeddingRandomMLP,
    FiLM,
    GlobalEmbeddingSimple,
    ProcessGlobalEmbeddingRandomMLP,
    SharedGlobalEmbeddingRandomMLP,
)
from .graph_layers import FeatureGNN


class _StagerBase(nn.Module):
    """
    TemporalDAG + GlobalEmbedding(FiLM) + PyG-based GNN combined temporal DAG model.

    Args:
        gnn_type: 'gcn', 'gat', or 'gatv2'
        gnn_hidden_dim: GNN intermediate dimension (None is the same as dag_output_dim)
        gnn_layers: Number of MessagePassing layers
        gnn_heads: Number of GATConv heads (used when gnn_type='gat')
        temporal_type: 'dag' (default), 'fcg', or 'fc'
        conditioning_type: 'film' (default), 'concat', or 'none'
    """

    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        adj_func: str = "sigmoid",
        gnn_type: str = "gcn",
        gnn_layers: int = 2,
        gnn_hidden_dim: Optional[int] = None,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.0,
        node_dim: Optional[int] = None,
        film_hidden_ratio: float = 2.0,
        use_ema_in_forward: bool = False,
        feature_residual: bool = False,
        film_pre_norm_type: Optional[str] = None,
        film_post_norm_type: Optional[str] = "ln",
        dag_use_bn: bool = False,
        feature_encoder_type: str = "multi_scale",
        temporal_type: str = "dag",
        conditioning_type: str = "film",
        tag_mixing: bool = True,
        nonmix_method: str = "per_linear",
        tgnn_layers: int = 2,
        tgnn_hidden_multiplier: int = 2,
        shift: int = 2,
        temporal_pool: str = "mean",
        ms_windows: Optional[Union[List[int], str]] = "5,10,20",
        feature_edge_layout: str = "legacy",
    ):
        super().__init__()

        # 1) Temporal graph backbone
        if temporal_type == 'dag':
            graph_type = 'dag'
        elif temporal_type == 'fcg' or temporal_type == 'fc':
            graph_type = 'fc'
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.dag = TemporalDAG(
            num_features, feature_dim, adj_func, node_dim,
            use_bn=dag_use_bn,
            feature_encoder_type=feature_encoder_type,
            tag_mixing=tag_mixing,
            nonmix_method=nonmix_method,
            tgnn_layers=tgnn_layers,
            tgnn_hidden_multiplier=tgnn_hidden_multiplier,
            shift=shift,
            temporal_pool=temporal_pool,
            ms_windows=ms_windows,
            graph_type=graph_type,
        )

        # TemporalDAG falls back to feature_dim when node_dim is omitted. Keep
        # the public model dimensions consistent with that effective value.
        self.dag_output_dim = self.dag.node_dim
        
        # Residual option
        self.feature_residual = feature_residual

        # 2) Conditioning (GlobalEmbedding, FiLM, Concat)
        if conditioning_type == 'none' or conditioning_type is None:
            self.conditioning_type = None
        else:
            self.conditioning_type = conditioning_type

        self.global_embedding = None
        self.film = None
        self.concat_mlp = None

        if self.conditioning_type is not None:
            # Global Embedding (EMA supported)
            self.global_embedding = GlobalEmbeddingSimple(num_features, self.dag_output_dim)
            self.use_ema_in_forward = use_ema_in_forward

            film_hidden = int(self.dag_output_dim * film_hidden_ratio)

            if self.conditioning_type == 'concat':
                self.concat_mlp = nn.Sequential(
                    nn.Linear(self.dag_output_dim * 2, film_hidden),
                    nn.GELU(),
                    nn.Linear(film_hidden, self.dag_output_dim),
                    nn.LayerNorm(self.dag_output_dim),
                )
            elif self.conditioning_type == 'film':
                self.film = FiLM(
                    cond_dim=self.dag_output_dim, 
                    target_dim=self.dag_output_dim, 
                    num_features=num_features, 
                    hidden=film_hidden, 
                    pre_norm_type=film_pre_norm_type,
                    post_norm_type=film_post_norm_type,
                )
            else:
                raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

        # 3) PyG GNN block
        gnn_hidden_dim = gnn_hidden_dim or self.dag_output_dim
        self.feature_gnn = FeatureGNN(
            num_nodes=num_features,
            in_dim=self.dag_output_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=self.dag_output_dim,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            heads=gnn_heads,
            dropout=gnn_dropout,
            feature_residual=self.feature_residual,
            edge_layout=feature_edge_layout,
        )

        # 4) Per-node head (predict the next time-step for each feature)
        self.fc = nn.Linear(self.dag_output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return:
            out: (B, F)              # next time-step prediction for each feature
        """
        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1).contiguous()

        # Temporal graph encoding: (B, F, D_dag)
        x_encoded = self.dag(x)

        # Conditioning
        if self.conditioning_type is not None:
            # Select global embedding
            S = self.global_embedding.get_for_forward(
                training=self.training,
                use_ema_in_forward=self.use_ema_in_forward,
            )  # (F, D_dag)

            if self.conditioning_type == 'concat':
                S_b = S.unsqueeze(0).expand(x_encoded.size(0), -1, -1)  # (B, F, D_dag)
                x_cat = torch.cat([x_encoded, S_b], dim=-1)
                x_mod = self.concat_mlp(x_cat)
            elif self.conditioning_type == 'film':
                x_mod = self.film(x_encoded, S)
            else:
                x_mod = x_encoded # Should not reach here
        else:
            x_mod = x_encoded

        # GNN propagation
        x_gnn = self.feature_gnn(x_mod)  # (B, F, D_out)

        # Head
        out = self.fc(x_gnn).squeeze(-1)  # (B, F)

        return out


class STAGER(_StagerBase):
    """
    STAGER model with RandomMLP global embeddings and auxiliary outputs.
    """
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        adj_func: str = "sigmoid",
        gnn_type: str = "gcn",
        gnn_layers: int = 2,
        gnn_hidden_dim: Optional[int] = None,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.0,
        node_dim: Optional[int] = None,
        film_hidden_ratio: float = 2.0,
        use_ema_in_forward: bool = False,
        feature_residual: bool = False,
        film_pre_norm_type: Optional[str] = None,
        film_post_norm_type: Optional[str] = "ln",
        dag_use_bn: bool = False,
        feature_encoder_type: str = "multi_scale",
        temporal_type: str = "dag",
        conditioning_type: str = "film",
        tag_mixing: bool = True,
        nonmix_method: str = "per_linear",
        tgnn_layers: int = 2,
        tgnn_hidden_multiplier: int = 2,
        shift: int = 2,
        # --- GlobalEmbedding(RandomMLP) related ---
        global_seed_dim: Optional[int] = None,
        global_mlp_hidden: Optional[int] = None,
        global_mlp_layers: int = 2,
        global_mlp_init: str = "xavier_uniform",
        global_mlp_out_norm: Optional[str] = None,
        # --- Aux classifier ---
        aux_target: str = "global",
        aux_classifier_hidden: Optional[int] = None,
        aux_classifier_layers: int = 2,
        aux_classifier_dropout: float = 0.0,
        temporal_pool: str = "mean",
        ms_windows: Optional[Union[List[int], str]] = "5,10,20",
        process_to_tag: Optional[List[int]] = None,
        feature_edge_layout: str = "legacy",
    ):
        self._embedding_variant = "per_tag"
        base_conditioning_type = conditioning_type
        if conditioning_type is not None:
            if conditioning_type.endswith("_shared"):
                self._embedding_variant = "shared"
                base_conditioning_type = conditioning_type[: -len("_shared")]
            elif conditioning_type.endswith("_process"):
                self._embedding_variant = "process"
                base_conditioning_type = conditioning_type[: -len("_process")]

        super().__init__(
            num_features=num_features,
            feature_dim=feature_dim,
            adj_func=adj_func,
            gnn_type=gnn_type,
            gnn_layers=gnn_layers,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_heads=gnn_heads,
            gnn_dropout=gnn_dropout,
            node_dim=node_dim,
            film_hidden_ratio=film_hidden_ratio,
            use_ema_in_forward=use_ema_in_forward,
            feature_residual=feature_residual,
            film_pre_norm_type=film_pre_norm_type,
            film_post_norm_type=film_post_norm_type,
            dag_use_bn=dag_use_bn,
            feature_encoder_type=feature_encoder_type,
            temporal_type=temporal_type,
            conditioning_type=base_conditioning_type,
            tag_mixing=tag_mixing,
            nonmix_method=nonmix_method,
            tgnn_layers=tgnn_layers,
            tgnn_hidden_multiplier=tgnn_hidden_multiplier,
            shift=shift,
            temporal_pool=temporal_pool,
            ms_windows=ms_windows,
            feature_edge_layout=feature_edge_layout,
        )

        # Keep this defined even when conditioning is disabled.
        self.use_ema_in_forward = use_ema_in_forward

        self.aux_target = aux_target
        self._process_to_tag = process_to_tag

        if self.conditioning_type is not None:
            emb_common = dict(
                num_tags=num_features,
                dim=self.dag_output_dim,
                seed_dim=global_seed_dim,
                hidden_dim=global_mlp_hidden,
                num_layers=global_mlp_layers,
                init=global_mlp_init,
                out_norm=global_mlp_out_norm,
            )
            if self._embedding_variant == "shared":
                self.global_embedding = SharedGlobalEmbeddingRandomMLP(**emb_common)
            elif self._embedding_variant == "process":
                if process_to_tag is None:
                    raise ValueError(
                        "conditioning_type ends with '_process' but process_to_tag is None. "
                        "Provide process_to_tag (List[int] of length num_features)."
                    )
                self.global_embedding = ProcessGlobalEmbeddingRandomMLP(
                    process_to_tag=process_to_tag, **emb_common
                )
            else:
                self.global_embedding = GlobalEmbeddingRandomMLP(**emb_common)

        num_aux_classes = (
            self.global_embedding.num_unique_profiles
            if self.global_embedding is not None
            else num_features
        )

        aux_input_dim = self.dag_output_dim
        if self.aux_target == "conditioning" and self.conditioning_type == "film":
            aux_input_dim = 2 * self.dag_output_dim

        aux_layers = max(1, int(aux_classifier_layers))
        aux_hidden = aux_classifier_hidden if aux_classifier_hidden is not None else max(16, 2 * aux_input_dim)
        if aux_layers == 1:
            self.global_classifier = nn.Linear(aux_input_dim, num_aux_classes)
        else:
            layers = [nn.Linear(aux_input_dim, aux_hidden), nn.GELU()]
            if aux_classifier_dropout > 0:
                layers.append(nn.Dropout(aux_classifier_dropout))
            for _ in range(aux_layers - 2):
                layers.extend([nn.Linear(aux_hidden, aux_hidden), nn.GELU()])
                if aux_classifier_dropout > 0:
                    layers.append(nn.Dropout(aux_classifier_dropout))
            layers.append(nn.Linear(aux_hidden, num_aux_classes))
            self.global_classifier = nn.Sequential(*layers)

    def get_global_embeddings(self, normalize: bool = False, eps: float = 1e-8) -> Optional[torch.Tensor]:
        if self.global_embedding is None:
            return None
        S = self.global_embedding.get_for_forward(
            training=self.training,
            use_ema_in_forward=self.use_ema_in_forward,
        )
        if normalize:
            S = F.normalize(S, dim=-1, eps=eps)
        return S

    def get_global_embedding_logits(
        self,
        S: Optional[torch.Tensor] = None,
        normalize: bool = False,
        eps: float = 1e-8,
    ) -> Optional[torch.Tensor]:
        if S is None:
            S = self.get_global_embeddings(normalize=normalize, eps=eps)
        elif normalize:
            S = F.normalize(S, dim=-1, eps=eps)
        if S is None:
            return None
        return self.global_classifier(S)

    @staticmethod
    def _l2_norm(t: Optional[torch.Tensor], eps: float = 1e-8) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return F.normalize(t, p=2, dim=-1, eps=eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        x: (B, T, F)
        return:
            out: (B, F)
            intermediates:
                - global: (N, D) — N=F(per-tag), N=P(process), or None(shared)
                - conditioning: (F, 2D) or None
                - local: (B, F, D)
        """
        x = x.permute(0, 2, 1).contiguous()

        x_encoded = self.dag(x)

        S = None
        S_aux = None
        cond_repr = None

        if self.conditioning_type is not None:
            S = self.global_embedding.get_for_forward(
                training=self.training,
                use_ema_in_forward=self.use_ema_in_forward,
            )  # (F, D_dag)
            S_aux = self.global_embedding.get_aux_profiles()  # (N, D) or None

            if self.conditioning_type == "concat":
                S_b = S.unsqueeze(0).expand(x_encoded.size(0), -1, -1)
                x_cat = torch.cat([x_encoded, S_b], dim=-1)
                x_mod = self.concat_mlp(x_cat)
            elif self.conditioning_type == "film":
                if self.aux_target == "conditioning":
                    x_mod, cond_repr = self.film(x_encoded, S, return_intermediate=True)
                else:
                    x_mod = self.film(x_encoded, S)
            else:
                x_mod = x_encoded
        else:
            x_mod = x_encoded

        x_gnn = self.feature_gnn(x_mod)
        out = self.fc(x_gnn).squeeze(-1)

        intermediates = {
            "global": self._l2_norm(S_aux),
            "conditioning": self._l2_norm(cond_repr),
            "local": self._l2_norm(x_encoded),
        }

        return out, intermediates


__all__ = ["STAGER"]
