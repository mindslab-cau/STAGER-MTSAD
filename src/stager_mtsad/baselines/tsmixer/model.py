# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified: the original TensorFlow forecasting model was
# adapted to PyTorch and constrained to all-feature next-point prediction.

"""PyTorch TSMixer model for multivariate next-point forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TSMixerModelError(ValueError):
    """Raised when a TSMixer model configuration or input is invalid."""


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"identity", "linear", "none"}:
        return nn.Identity()
    raise TSMixerModelError(
        "activation must be 'relu', 'gelu', or 'identity'"
    )


class ResidualBlock(nn.Module):
    """One temporal-mixing and feature-mixing residual block."""

    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        *,
        norm_type: str,
        activation: str,
        dropout: float,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        if norm_type not in {"L", "B"}:
            raise TSMixerModelError("norm_type must be 'L' or 'B'")

        self.norm_type = norm_type
        if norm_type == "L":
            self.normalization = nn.LayerNorm(
                (sequence_length, feature_count)
            )
        else:
            self.normalization = nn.BatchNorm1d(feature_count)

        self.temporal_linear = nn.Linear(sequence_length, sequence_length)
        self.feature_linear_1 = nn.Linear(feature_count, feedforward_dim)
        self.feature_linear_2 = nn.Linear(feedforward_dim, feature_count)
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "L":
            return self.normalization(values)
        values = values.transpose(1, 2)
        values = self.normalization(values)
        return values.transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Mix an input shaped ``[batch, time, feature]``."""

        residual = values
        values = self._normalize(values)
        values = values.transpose(1, 2)
        values = self.temporal_linear(values)
        values = self.activation(values)
        values = values.transpose(1, 2)
        values = self.dropout(values)
        values = values + residual

        residual = values
        values = self._normalize(values)
        values = self.feature_linear_1(values)
        values = self.activation(values)
        values = self.dropout(values)
        values = self.feature_linear_2(values)
        values = self.dropout(values)
        return values + residual


class TSMixer(nn.Module):
    """TSMixer constrained to one-step prediction of every input feature."""

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        *,
        pred_len: int = 1,
        norm_type: str = "L",
        activation: str = "relu",
        n_block: int = 2,
        dropout: float = 0.1,
        ff_dim: int = 64,
        target_slice: None = None,
    ) -> None:
        super().__init__()
        if input_len < 1:
            raise TSMixerModelError("input_len must be positive")
        if input_dim < 1:
            raise TSMixerModelError("input_dim must be positive")
        if pred_len != 1:
            raise TSMixerModelError(
                "the anomaly-detection adapter requires pred_len=1"
            )
        if target_slice is not None:
            raise TSMixerModelError(
                "target_slice must be None so every feature is predicted"
            )
        if n_block < 1:
            raise TSMixerModelError("n_block must be positive")
        if ff_dim < 1:
            raise TSMixerModelError("ff_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise TSMixerModelError("dropout must be in [0, 1)")

        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.pred_len = int(pred_len)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    self.input_len,
                    self.input_dim,
                    norm_type=norm_type,
                    activation=activation,
                    dropout=float(dropout),
                    feedforward_dim=int(ff_dim),
                )
                for _ in range(int(n_block))
            ]
        )
        self.projection = nn.Linear(self.input_len, self.pred_len)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predict the first point after each input window.

        Args:
            values: Float tensor shaped ``[batch, input_len, input_dim]``.

        Returns:
            A tensor shaped ``[batch, input_dim]``.
        """

        if values.ndim != 3:
            raise TSMixerModelError(
                "input must have shape [batch, time, feature]"
            )
        if values.shape[1:] != (self.input_len, self.input_dim):
            raise TSMixerModelError(
                "input time and feature dimensions do not match the model"
            )

        for block in self.residual_blocks:
            values = block(values)

        values = values.transpose(1, 2)
        values = self.projection(values)
        return values.squeeze(-1)


def build_model(
    input_shape: Sequence[int],
    pred_len: int = 1,
    norm_type: str = "L",
    activation: str = "relu",
    n_block: int = 2,
    dropout: float = 0.1,
    ff_dim: int = 64,
    target_slice: None = None,
) -> TSMixer:
    """Build the locally adapted PyTorch TSMixer model."""

    if len(input_shape) != 2:
        raise TSMixerModelError(
            "input_shape must contain sequence length and feature count"
        )
    sequence_length, feature_count = input_shape
    return TSMixer(
        input_len=int(sequence_length),
        input_dim=int(feature_count),
        pred_len=pred_len,
        norm_type=norm_type,
        activation=activation,
        n_block=n_block,
        dropout=dropout,
        ff_dim=ff_dim,
        target_slice=target_slice,
    )
