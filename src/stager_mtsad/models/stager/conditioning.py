import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def build_norm(
    norm_type: Optional[str],
    target_dim: int,
    num_features: int,
) -> Optional[nn.Module]:
    """
    norm_type:
      - None / 'none' : no norm
      - 'ln'          : LayerNorm over last dim (D)
      - 'ln2d'        : LayerNorm over (F, D) jointly
    """
    if norm_type is None or norm_type == "none":
        return None
    if norm_type == "ln":
        return nn.LayerNorm(target_dim)
    if norm_type == "ln2d":
        return nn.LayerNorm((num_features, target_dim))
    raise ValueError(f"Invalid norm_type={norm_type}. Choose from None/'none', 'ln', 'ln2d'.")


class GlobalEmbeddingSimple(nn.Module):
    """
    Maintains the global embedding simply as a 'learnable parameter (S_param)' and an 'EMA buffer (S_ema)'.
    - Which one to use in forward is determined by get_for_forward() (default is S_param).
    - After each batch (step), the trainer should call ema_update(alpha) to update S_ema.
    """
    def __init__(self, num_tags: int, dim: int):
        super().__init__()
        self.num_tags, self.dim = num_tags, dim

        # Learnable parameter
        self.S_param = nn.Parameter(torch.empty(num_tags, dim))
        nn.init.normal_(self.S_param, std=0.02)

        # EMA buffer: initially copy S_param
        self.register_buffer("S_ema", self.S_param.data.clone())

    def get_for_forward(self, training: bool, use_ema_in_forward: bool = False):
        """
        Select the global vector to use in forward.
        - Normally, use S_param during training (connected gradient),
          If stability is very important, use EMA with use_ema_in_forward=True.
        - It is recommended to use EMA in evaluation/inference.
        """
        if training:
            return self.S_ema if use_ema_in_forward else self.S_param
        else:
            return self.S_ema  # It is recommended to use EMA in evaluation/inference.

    @torch.no_grad()
    def ema_update(self, alpha: float):
        """
        Call after each batch (step): S_ema ← (1-α)·S_ema + α·S_param
        """
        self.S_ema.mul_(1 - alpha).add_(alpha * self.S_param)


class GlobalEmbeddingRandomMLP(nn.Module):
    """
    Random fixed seed vector -> MLP -> global embedding.
    - The seed vector itself is not learned (buffer).
    - The only trainable thing is the MLP parameters.
    """
    def __init__(
        self,
        num_tags: int,
        dim: int,
        seed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        init: str = "xavier_uniform",
        out_norm: Optional[str] = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.dim = dim
        self.seed_dim = seed_dim if seed_dim is not None else dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(16, 2 * dim)
        self.num_layers = max(1, int(num_layers))
        self.out_norm = out_norm if out_norm is not None else "none"

        seed = torch.empty(num_tags, self.seed_dim)
        self._init_seed(seed, init)
        self.register_buffer("seed_vectors", seed)

        self.mlp = self._build_mlp()

    def _init_seed(self, tensor: torch.Tensor, init: str):
        if init == "xavier_uniform":
            nn.init.xavier_uniform_(tensor)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(tensor)
        elif init == "normal":
            nn.init.normal_(tensor, mean=0.0, std=0.02)
        elif init == "uniform":
            nn.init.uniform_(tensor, a=-0.05, b=0.05)
        else:
            raise ValueError(
                f"Invalid init='{init}'. Choose from xavier_uniform/xavier_normal/normal/uniform."
            )

    def _build_mlp(self) -> nn.Module:
        if self.num_layers == 1:
            return nn.Linear(self.seed_dim, self.dim)

        layers = [
            nn.Linear(self.seed_dim, self.hidden_dim),
            nn.GELU(),
        ]
        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()])
        layers.append(nn.Linear(self.hidden_dim, self.dim))
        return nn.Sequential(*layers)

    @property
    def num_unique_profiles(self) -> int:
        return self.num_tags

    def _apply_out_norm(self, out: torch.Tensor) -> torch.Tensor:
        if self.out_norm == "l2":
            return F.normalize(out, dim=-1, eps=1e-8)
        if self.out_norm not in ("none", None):
            raise ValueError("out_norm must be 'none' or 'l2'")
        return out

    def get_for_forward(self, training: bool, use_ema_in_forward: bool = False):
        """
        Provides the same interface as GlobalEmbeddingSimple.
   
        """
        return self._apply_out_norm(self.mlp(self.seed_vectors))

    def get_aux_profiles(self) -> Optional[torch.Tensor]:
        """For aux loss: return the tag-wise independent profiles (F, D) as is."""
        return self._apply_out_norm(self.mlp(self.seed_vectors))


class SharedGlobalEmbeddingRandomMLP(nn.Module):
    """
    Single seed vector (1, seed_dim) -> MLP -> (1, D) -> expand(F, D).
    All tags share the same global profile, and the gradient only flows through the (1, D) path.
    """
    def __init__(
        self,
        num_tags: int,
        dim: int,
        seed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        init: str = "xavier_uniform",
        out_norm: Optional[str] = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.dim = dim
        self.seed_dim = seed_dim if seed_dim is not None else dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(16, 2 * dim)
        self.num_layers = max(1, int(num_layers))
        self.out_norm = out_norm if out_norm is not None else "none"

        seed = torch.empty(1, self.seed_dim)
        self._init_seed(seed, init)
        self.register_buffer("seed_vector", seed)

        self.mlp = self._build_mlp()

    def _init_seed(self, tensor: torch.Tensor, init: str):
        if init == "xavier_uniform":
            nn.init.xavier_uniform_(tensor)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(tensor)
        elif init == "normal":
            nn.init.normal_(tensor, mean=0.0, std=0.02)
        elif init == "uniform":
            nn.init.uniform_(tensor, a=-0.05, b=0.05)
        else:
            raise ValueError(
                f"Invalid init='{init}'. Choose from xavier_uniform/xavier_normal/normal/uniform."
            )

    def _build_mlp(self) -> nn.Module:
        if self.num_layers == 1:
            return nn.Linear(self.seed_dim, self.dim)

        layers = [
            nn.Linear(self.seed_dim, self.hidden_dim),
            nn.GELU(),
        ]
        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()])
        layers.append(nn.Linear(self.hidden_dim, self.dim))
        return nn.Sequential(*layers)

    @property
    def num_unique_profiles(self) -> int:
        return 1

    def _apply_out_norm(self, out: torch.Tensor) -> torch.Tensor:
        if self.out_norm == "l2":
            return F.normalize(out, dim=-1, eps=1e-8)
        if self.out_norm not in ("none", None):
            raise ValueError("out_norm must be 'none' or 'l2'")
        return out

    def get_for_forward(self, training: bool, use_ema_in_forward: bool = False):
        """(1, D) -> expand(num_tags, D)"""
        return self._apply_out_norm(self.mlp(self.seed_vector)).expand(self.num_tags, -1)

    def get_aux_profiles(self) -> Optional[torch.Tensor]:
        """Single profile → aux loss not required."""
   
        return None


class ProcessGlobalEmbeddingRandomMLP(nn.Module):
    """
    Process-wise shared profile: (P, seed_dim) -> MLP -> (P, D) -> expand to (F, D).
    Tags in the same process share the same profile.
    aux loss is only applied between the process representative profiles (P, D).
    """
    def __init__(
        self,
        num_tags: int,
        dim: int,
        process_to_tag: List[int],
        seed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        init: str = "xavier_uniform",
        out_norm: Optional[str] = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.dim = dim
        self.seed_dim = seed_dim if seed_dim is not None else dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(16, 2 * dim)
        self.num_layers = max(1, int(num_layers))
        self.out_norm = out_norm if out_norm is not None else "none"

        p2t = torch.tensor(process_to_tag, dtype=torch.long)
        assert p2t.size(0) == num_tags, (
            f"process_to_tag length ({p2t.size(0)}) != num_tags ({num_tags})"
        )
        self.register_buffer("process_to_tag", p2t)
        self._num_processes = int(p2t.max().item()) + 1

        seed = torch.empty(self._num_processes, self.seed_dim)
        self._init_seed(seed, init)
        self.register_buffer("seed_vectors", seed)

        self.mlp = self._build_mlp()

    @property
    def num_unique_profiles(self) -> int:
        return self._num_processes

    def _init_seed(self, tensor: torch.Tensor, init: str):
        if init == "xavier_uniform":
            nn.init.xavier_uniform_(tensor)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(tensor)
        elif init == "normal":
            nn.init.normal_(tensor, mean=0.0, std=0.02)
        elif init == "uniform":
            nn.init.uniform_(tensor, a=-0.05, b=0.05)
        else:
            raise ValueError(
                f"Invalid init='{init}'. Choose from xavier_uniform/xavier_normal/normal/uniform."
            )

    def _build_mlp(self) -> nn.Module:
        if self.num_layers == 1:
            return nn.Linear(self.seed_dim, self.dim)
        layers = [nn.Linear(self.seed_dim, self.hidden_dim), nn.GELU()]
        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()])
        layers.append(nn.Linear(self.hidden_dim, self.dim))
        return nn.Sequential(*layers)

    def _apply_out_norm(self, out: torch.Tensor) -> torch.Tensor:
        if self.out_norm == "l2":
            return F.normalize(out, dim=-1, eps=1e-8)
        if self.out_norm not in ("none", None):
            raise ValueError("out_norm must be 'none' or 'l2'")
        return out

    def get_for_forward(self, training: bool, use_ema_in_forward: bool = False):
        """(P, D) -> index by process_to_tag -> (F, D)"""
        P_emb = self._apply_out_norm(self.mlp(self.seed_vectors))  # (P, D)
        return P_emb[self.process_to_tag]                          # (F, D)

    def get_aux_profiles(self) -> Optional[torch.Tensor]:
        """For aux loss: return the process representative profiles (P, D)."""
        return self._apply_out_norm(self.mlp(self.seed_vectors))   # (P, D)


class FiLM(nn.Module):
    """
    FiLM: feature-wise linear modulation

    x   : (B, F, D)   target features (e.g., DAG output)
    cond: (F, C)      tag-wise conditioning (e.g., global profile)

    Options:
      - pre_norm_type  : None/'none'/'ln'/'ln2d' (default: None)
      - post_norm_type : None/'none'/'ln'/'ln2d' (default: 'ln')
    """
    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        num_features: int,
        hidden: Optional[int] = None,
        pre_norm_type: Optional[str] = None,     # default: None (no pre-norm)
        post_norm_type: Optional[str] = "ln",    # default: ln (post-norm)
        norm_type: Optional[str] = None,         # deprecated (for backward compatibility)
    ):
        super().__init__()
        h = hidden or (2 * target_dim)

        # Backward compatibility: if norm_type is provided, it overrides post_norm_type
        if norm_type is not None:
            post_norm_type = norm_type

        self.pre_norm = build_norm(pre_norm_type, target_dim, num_features)
        self.post_norm = build_norm(post_norm_type, target_dim, num_features)

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, h),
            nn.GELU(),
            nn.Linear(h, 2 * target_dim),  # -> [gamma | beta]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        return_intermediate: bool = False,
    ):
        """
        x   : (B, F, D)
        cond: (F, C)
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        gb = self.mlp(cond)                     # (F, 2D)
        gamma_raw, beta_raw = gb.chunk(2, dim=-1)  # (F, D), (F, D)

        gamma = gamma_raw.unsqueeze(0).expand_as(x) # (B, F, D)
        beta  = beta_raw.unsqueeze(0).expand_as(x)  # (B, F, D)

        out = gamma * x + beta                  # (B, F, D)

        if self.post_norm is not None:
            out = self.post_norm(out)

        if return_intermediate:
            gamma_norm = F.normalize(gamma_raw, dim=-1, eps=1e-8)
            beta_norm = F.normalize(beta_raw, dim=-1, eps=1e-8)
            gb_norm = torch.cat([gamma_norm, beta_norm], dim=-1)
            return out, gb_norm
        return out
