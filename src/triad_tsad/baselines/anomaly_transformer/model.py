"""Anomaly Transformer model adapted for portable TriAD baseline runs."""

from __future__ import annotations

import math
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        positional_encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        positional_encoding.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        divisor = (
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        ).exp()

        positional_encoding[:, 0::2] = torch.sin(position * divisor)
        positional_encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", positional_encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : inputs.size(1)]


class TokenEmbedding(nn.Module):
    """Circular temporal convolution used by the legacy implementation."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        # Keep the historical attribute name so retained model state keys stay
        # compatible with the source architecture.
        self.tokenConv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.tokenConv.weight,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(inputs.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    """Value and positional embeddings."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(input_dim=input_dim, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.value_embedding(inputs) + self.position_embedding(inputs)
        return self.dropout(embedded)


class TriangularCausalMask:
    """Upper-triangular attention mask."""

    def __init__(self, batch_size: int, length: int, device: torch.device) -> None:
        shape = (batch_size, 1, length, length)
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(shape, dtype=torch.bool, device=device),
                diagonal=1,
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class AnomalyAttention(nn.Module):
    """Attention with a learned Gaussian prior association."""

    def __init__(
        self,
        window_size: int,
        mask_flag: bool = True,
        scale: float | None = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        positions = torch.arange(window_size, dtype=torch.float32)
        distances = torch.abs(positions[:, None] - positions[None, :])
        # The source code allocated this tensor on CUDA directly. A
        # non-persistent buffer preserves checkpoint keys while following the
        # model to the selected device.
        self.register_buffer("distances", distances, persistent=False)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        sigma: torch.Tensor,
        attention_mask: TriangularCausalMask | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch_size, query_length, _, embedding_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(embedding_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attention_mask is None:
                attention_mask = TriangularCausalMask(
                    batch_size,
                    query_length,
                    device=queries.device,
                )
            scores.masked_fill_(attention_mask.mask, -torch.inf)
        attention = scale * scores

        sigma = sigma.transpose(1, 2)
        effective_window_size = attention.shape[-1]
        if effective_window_size != self.window_size:
            raise ValueError(
                "input sequence length must match the configured window_size "
                f"({self.window_size}); received {effective_window_size}"
            )

        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, effective_window_size)

        distances = self.distances.to(dtype=sigma.dtype)
        prior_distance = distances.unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0],
            sigma.shape[1],
            1,
            1,
        )
        prior = (
            1.0
            / (math.sqrt(2 * math.pi) * sigma)
            * torch.exp(-(prior_distance**2) / 2 / (sigma**2))
        )

        series = self.dropout(torch.softmax(attention, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return output.contiguous(), series, prior, sigma
        return output.contiguous(), None, None, None


class AttentionLayer(nn.Module):
    """Multi-head projections around anomaly attention."""

    def __init__(
        self,
        attention: AnomalyAttention,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads

        # This norm is unused in the historical forward pass but is retained
        # for state-dict compatibility.
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: TriangularCausalMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length, _ = queries.shape
        _, source_length, _ = keys.shape
        heads = self.n_heads

        projected_queries = self.query_projection(queries).view(
            batch_size,
            query_length,
            heads,
            -1,
        )
        projected_keys = self.key_projection(keys).view(
            batch_size,
            source_length,
            heads,
            -1,
        )
        projected_values = self.value_projection(values).view(
            batch_size,
            source_length,
            heads,
            -1,
        )
        sigma = self.sigma_projection(queries).view(
            batch_size,
            query_length,
            heads,
        )

        output, series, prior, sigma = self.inner_attention(
            projected_queries,
            projected_keys,
            projected_values,
            sigma,
            attention_mask,
        )
        if series is None or prior is None or sigma is None:
            raise RuntimeError(
                "Anomaly Transformer training and scoring require "
                "output_attention=True"
            )
        output = output.view(batch_size, query_length, -1)
        return self.out_projection(output), series, prior, sigma


class EncoderLayer(nn.Module):
    """One anomaly-attention encoder block."""

    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: TriangularCausalMask | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        new_inputs, series, prior, sigma = self.attention(
            inputs,
            inputs,
            inputs,
            attention_mask=attention_mask,
        )
        inputs = inputs + self.dropout(new_inputs)
        residual = inputs = self.norm1(inputs)
        residual = self.dropout(
            self.activation(self.conv1(residual.transpose(-1, 1)))
        )
        residual = self.dropout(self.conv2(residual).transpose(-1, 1))
        return self.norm2(inputs + residual), series, prior, sigma


class Encoder(nn.Module):
    """Stacked anomaly-attention encoder."""

    def __init__(
        self,
        attention_layers: list[EncoderLayer],
        norm_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Keep the historical name for checkpoint compatibility.
        self.attn_layers = nn.ModuleList(attention_layers)
        self.norm = norm_layer

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: TriangularCausalMask | None = None,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        series_list: list[torch.Tensor] = []
        prior_list: list[torch.Tensor] = []
        sigma_list: list[torch.Tensor] = []

        for attention_layer in self.attn_layers:
            inputs, series, prior, sigma = attention_layer(
                inputs,
                attention_mask=attention_mask,
            )
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            inputs = self.norm(inputs)
        return inputs, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    """Reconstruction model used by the Anomaly Transformer baseline.

    ``output_attention=False`` is intentionally rejected because both the
    minimax training loss and legacy anomaly score require association tensors.
    The adapted source's false branch was not executable by its attention
    wrapper.
    """

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        output_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = True,
    ) -> None:
        super().__init__()
        if not output_attention:
            raise ValueError(
                "output_attention must be True for Anomaly Transformer "
                "training and scoring"
            )
        if window_size < 1:
            raise ValueError("window_size must be at least one")

        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_attention = output_attention
        self.embedding = DataEmbedding(input_dim, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            window_size,
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=True,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, output_dim, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        encoded = self.embedding(inputs)
        encoded, series, prior, sigmas = self.encoder(encoded)
        return self.projection(encoded), series, prior, sigmas
