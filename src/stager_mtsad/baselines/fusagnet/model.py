"""Local FuSAGNet model adaptation.

This module is intentionally not imported by the baseline package until the
model is requested. PyTorch Geometric remains an optional heavy dependency.
"""

from __future__ import annotations

from random import uniform
from typing import NamedTuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Linear, Parameter
except ImportError as exc:  # pragma: no cover - exercised in minimal environments
    raise ImportError("FuSAGNet requires PyTorch") from exc

try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn.inits import glorot, zeros
    from torch_geometric.utils import (
        add_self_loops,
        remove_self_loops,
        softmax,
    )
except ImportError as exc:  # pragma: no cover - dependency-aware tests cover import
    raise ImportError(
        "FuSAGNet requires the optional PyTorch Geometric dependency"
    ) from exc

from .config import FeatureLayout, FuSAGNetModelConfig


def get_batch_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    node_count: int,
) -> torch.Tensor:
    """Repeat one graph with disjoint node offsets for a mini-batch."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge]")
    edge_count = edge_index.shape[1]
    repeated = edge_index.detach().clone().repeat(1, batch_size).contiguous()
    offsets = (
        torch.arange(batch_size, device=repeated.device, dtype=repeated.dtype)
        .repeat_interleave(edge_count)
        .mul(node_count)
    )
    repeated += offsets.unsqueeze(0)
    return repeated.long()


class GraphLayer(MessagePassing):
    """Embedding-aware graph attention layer used by FuSAGNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self._attention: torch.Tensor | None = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.att_i = Parameter(torch.empty(1, heads, out_channels))
        self.att_j = Parameter(torch.empty(1, heads, out_channels))
        self.att_em_i = Parameter(torch.empty(1, heads, out_channels))
        self.att_em_j = Parameter(torch.empty(1, heads, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        embedding: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if embedding is None:
            raise ValueError("FuSAGNet GraphLayer requires node embeddings")
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        transformed = self.lin(x)
        out = self.propagate(
            edge_index=edge_index,
            x=(transformed, transformed),
            embedding=embedding,
            edges=edge_index,
            return_attention_weights=return_attention_weights,
        )

        if self.concat and self.heads > 1:
            out = out.view(-1, self.heads * self.out_channels)
        elif not self.concat and self.heads > 1:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias

        if not return_attention_weights:
            return out
        attention, self._attention = self._attention, None
        if attention is None:
            raise RuntimeError("attention weights were not produced")
        return out, (edge_index, attention)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index_i: torch.Tensor,
        size_i: int | None,
        embedding: torch.Tensor,
        edges: torch.Tensor,
        return_attention_weights: bool,
    ) -> torch.Tensor:
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        embedding_i = embedding[edge_index_i]
        embedding_j = embedding[edges[0]]
        embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
        embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

        key_i = torch.cat((x_i, embedding_i), dim=-1)
        key_j = torch.cat((x_j, embedding_j), dim=-1)
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        attended_i = torch.einsum("nhd,mhd->nhd", key_i, cat_att_i)
        attended_j = torch.einsum("nhd,mhd->nhd", key_j, cat_att_j)
        alpha = F.leaky_relu(
            attended_i.sum(dim=-1) + attended_j.sum(dim=-1),
            self.negative_slope,
        )
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if return_attention_weights:
            self._attention = alpha

        result = alpha.unsqueeze(-1) * x_j
        if self.heads == 1:
            result = result.squeeze(1)
        return result


class OutputLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layer_count: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index in range(layer_count):
            if index == layer_count - 1:
                modules.append(
                    nn.Linear(input_size if layer_count == 1 else hidden_size, 1)
                )
            else:
                modules.extend(
                    [
                        nn.Linear(input_size if index == 0 else hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                    ]
                )
        self.mlp = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for module in self.mlp:
            if isinstance(module, nn.BatchNorm1d):
                out = module(out.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                out = module(out)
        return out


def _interpolated_layer_sizes(
    outer_size: int,
    inner_size: int,
    layer_count: int,
) -> list[int]:
    return [outer_size] + [
        outer_size
        - (index + 1) * (outer_size - inner_size) // layer_count
        for index in range(layer_count)
    ]


class SparseEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        layer_count: int,
        *,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        sizes = _interpolated_layer_sizes(input_size, latent_size, layer_count)
        self.layers = nn.ModuleList(
            nn.Linear(sizes[index], sizes[index + 1])
            for index in range(len(sizes) - 1)
        )
        self.batch_norms = (
            nn.ModuleList(nn.BatchNorm1d(layer.out_features) for layer in self.layers)
            if use_batch_norm
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)
        feature_maps: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[index](x)
            x = torch.sigmoid(x)
            feature_maps.append(x)
        return x, feature_maps


class SparseDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        layer_count: int,
        *,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        sizes = list(
            reversed(
                _interpolated_layer_sizes(output_size, latent_size, layer_count)
            )
        )
        self.layers = nn.ModuleList(
            nn.Linear(sizes[index], sizes[index + 1])
            for index in range(len(sizes) - 1)
        )
        self.batch_norms = (
            nn.ModuleList(nn.BatchNorm1d(layer.out_features) for layer in self.layers)
            if use_batch_norm
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feature_maps: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[index](x)
            x = torch.sigmoid(x)
            feature_maps.append(x)
        return x, feature_maps


class GNNLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.gnn = GraphLayer(
            in_channels=input_size,
            out_channels=output_size,
            heads=1,
            concat=False,
        )
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.gnn(
            x,
            edge_index,
            embedding,
            return_attention_weights=True,
        )
        return self.activation(self.batch_norm(out))


def kl_divergence(rho: torch.Tensor, rho_hat: torch.Tensor) -> torch.Tensor:
    """KL divergence term used by the sparse autoencoder."""

    epsilon = 1e-8
    rho = rho.clamp(epsilon, 1 - epsilon)
    rho_hat = rho_hat.clamp(epsilon, 1 - epsilon)
    result = rho * torch.log(rho / rho_hat)
    result += (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return result.sum()


class FuSAGNetOutput(NamedTuple):
    forecast: torch.Tensor
    reconstruction: torch.Tensor
    latent_by_feature: torch.Tensor
    mu: None
    log_variance: None
    encoder_feature_maps: list[torch.Tensor]
    decoder_feature_maps: list[torch.Tensor]
    repeated_embeddings: torch.Tensor
    process_ids: torch.Tensor
    rho: torch.Tensor
    rho_hat: torch.Tensor


class FuSAGNet(nn.Module):
    """FuSAGNet with explicit feature-to-process index mapping."""

    def __init__(
        self,
        *,
        num_features: int,
        process_groups: tuple[tuple[int, ...], ...],
        embedding_dim: int = 16,
        output_hidden_dim: int = 16,
        window_size: int = 16,
        output_layers: int = 1,
        topk: int = 15,
        latent_size: int = 16,
        autoencoder_layers: int = 2,
        alpha: float = 0.5,
        beta: float = 1.0,
        process_rnn_layers: int = 3,
        dropout: float = 0.2,
        edge_set_count: int = 1,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even")
        if topk <= 0 or topk > num_features:
            raise ValueError("topk must be between one and num_features")
        if edge_set_count <= 0:
            raise ValueError("edge_set_count must be positive")

        flattened = [index for group in process_groups for index in group]
        if sorted(flattened) != list(range(num_features)):
            raise ValueError(
                "process_groups must assign every feature index exactly once"
            )

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.topk = topk
        self.latent_size = latent_size
        self.alpha = alpha
        self.beta = beta
        self.edge_set_count = edge_set_count

        self.process_embeddings = nn.ModuleList(
            nn.Embedding(len(indices), embedding_dim) for indices in process_groups
        )
        self.process_rnns = nn.ModuleList(
            nn.GRU(
                embedding_dim,
                embedding_dim // 2,
                bidirectional=True,
                num_layers=process_rnn_layers,
                dropout=dropout if process_rnn_layers > 1 else 0.0,
            )
            for _ in process_groups
        )
        for group_index, indices in enumerate(process_groups):
            self.register_buffer(
                f"_process_indices_{group_index}",
                torch.tensor(indices, dtype=torch.long),
                persistent=True,
            )

        feature_process_ids = torch.empty(num_features, dtype=torch.long)
        for process_index, indices in enumerate(process_groups):
            feature_process_ids[list(indices)] = process_index
        self.register_buffer(
            "_feature_process_ids", feature_process_ids, persistent=True
        )

        flattened_size = num_features * window_size
        latent_flattened_size = num_features * latent_size
        self.encoder = SparseEncoder(
            flattened_size,
            latent_flattened_size,
            autoencoder_layers,
            use_batch_norm=True,
        )
        self.decoder = SparseDecoder(
            latent_flattened_size,
            flattened_size,
            autoencoder_layers,
            use_batch_norm=True,
        )
        self.gnn_layers = nn.ModuleList(
            GNNLayer(latent_size, embedding_dim)
            for _ in range(edge_set_count)
        )

        graph_output_size = embedding_dim * edge_set_count
        self.output_batch_norm = nn.BatchNorm1d(graph_output_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = OutputLayer(
            graph_output_size,
            output_hidden_dim,
            output_layers,
        )
        self.reset_parameters()

    @classmethod
    def from_config(
        cls,
        config: FuSAGNetModelConfig,
        layout: FeatureLayout,
    ) -> FuSAGNet:
        return cls(
            num_features=layout.num_features,
            process_groups=tuple(
                group.feature_indices for group in layout.process_groups
            ),
            embedding_dim=config.embedding_dim,
            output_hidden_dim=config.output_hidden_dim,
            window_size=config.window_size,
            output_layers=config.output_layers,
            topk=config.topk,
            latent_size=config.latent_size,
            autoencoder_layers=config.autoencoder_layers,
            alpha=config.alpha,
            beta=config.beta,
            process_rnn_layers=config.process_rnn_layers,
            dropout=config.dropout,
            edge_set_count=config.edge_set_count,
        )

    def reset_parameters(self) -> None:
        for embedding in self.process_embeddings:
            nn.init.kaiming_uniform_(embedding.weight, a=0.0)

    def _ordered_embeddings(self, device: torch.device) -> torch.Tensor:
        group_indices: list[torch.Tensor] = []
        group_embeddings: list[torch.Tensor] = []
        for group_index, (embedding, rnn) in enumerate(
            zip(self.process_embeddings, self.process_rnns, strict=True)
        ):
            indices = getattr(self, f"_process_indices_{group_index}").to(device)
            local_ids = torch.arange(
                embedding.num_embeddings,
                device=device,
                dtype=torch.long,
            )
            embedded = embedding(local_ids).unsqueeze(0)
            embedded, _ = rnn(embedded)
            group_indices.append(indices)
            group_embeddings.append(embedded.squeeze(0))

        all_indices = torch.cat(group_indices)
        all_embeddings = torch.cat(group_embeddings)
        return all_embeddings[torch.argsort(all_indices)]

    def forward(self, data: torch.Tensor) -> FuSAGNetOutput:
        if data.ndim != 3:
            raise ValueError("data must have shape [batch, time, feature]")
        batch_size, window_size, feature_count = data.shape
        if window_size != self.window_size:
            raise ValueError(
                f"expected window size {self.window_size}, received {window_size}"
            )
        if feature_count != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features, received {feature_count}"
            )

        feature_major = data.detach().clone().permute(0, 2, 1).contiguous()
        latent, encoder_maps = self.encoder(feature_major)
        latent_by_feature = (
            latent.view(batch_size, self.num_features, self.latent_size)
            .detach()
            .clone()
        )

        reconstruction, decoder_maps = self.decoder(latent)
        reconstruction = (
            reconstruction.view_as(feature_major)
            .permute(0, 2, 1)
            .contiguous()
        )

        rho = torch.tensor(
            [uniform(1e-5, 1e-2) for _ in range(latent.size(1))],
            device=data.device,
            dtype=latent.dtype,
        ).unsqueeze(0)
        rho_hat = latent.mean(dim=0, keepdim=True)
        latent_nodes = latent.view(-1, self.latent_size).contiguous()

        embeddings = self._ordered_embeddings(data.device)
        repeated_embeddings = embeddings.repeat(batch_size, 1)
        similarity = torch.matmul(embeddings, embeddings.T)
        norm_product = torch.matmul(
            embeddings.norm(dim=-1).view(-1, 1),
            embeddings.norm(dim=-1).view(1, -1),
        ).clamp_min(torch.finfo(embeddings.dtype).eps)
        similarity = similarity / norm_product
        neighbors = torch.topk(similarity, self.topk, dim=-1).indices

        targets = (
            torch.arange(self.num_features, device=data.device)
            .unsqueeze(1)
            .repeat(1, self.topk)
            .flatten()
        )
        sources = neighbors.flatten()
        edge_index = torch.stack((sources, targets), dim=0)
        batch_edge_index = get_batch_edge_index(
            edge_index,
            batch_size,
            self.num_features,
        )

        graph_outputs = [
            layer(latent_nodes, batch_edge_index, repeated_embeddings)
            for layer in self.gnn_layers
        ]
        graph_output = torch.cat(graph_outputs, dim=1)
        graph_output = graph_output.view(batch_size, self.num_features, -1)
        embedding_gate = embeddings.repeat(1, self.edge_set_count)
        graph_output = graph_output * embedding_gate

        graph_output = graph_output.permute(0, 2, 1)
        graph_output = F.relu(self.output_batch_norm(graph_output))
        graph_output = graph_output.permute(0, 2, 1)
        graph_output = self.output_dropout(graph_output)
        forecast = self.output_layer(graph_output).view(
            batch_size, self.num_features
        )

        process_ids = self._feature_process_ids.repeat(batch_size)
        return FuSAGNetOutput(
            forecast=forecast,
            reconstruction=reconstruction,
            latent_by_feature=latent_by_feature,
            mu=None,
            log_variance=None,
            encoder_feature_maps=encoder_maps[:-1],
            decoder_feature_maps=list(reversed(decoder_maps[:-1])),
            repeated_embeddings=repeated_embeddings,
            process_ids=process_ids,
            rho=rho,
            rho_hat=rho_hat,
        )

    def compute_loss(
        self,
        output: FuSAGNetOutput,
        targets: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forecast_loss = torch.sqrt(F.mse_loss(output.forecast, targets))
        reconstruction_mse = F.mse_loss(output.reconstruction, inputs)
        sparse_loss = kl_divergence(output.rho, output.rho_hat)
        reconstruction_loss = reconstruction_mse + self.beta * sparse_loss
        total = (
            self.alpha * forecast_loss
            + (1.0 - self.alpha) * reconstruction_loss
        )
        return total, forecast_loss, reconstruction_loss
