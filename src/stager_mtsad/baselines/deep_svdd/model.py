"""LeNet-style time-series networks used by the Deep SVDD adapter.

The architecture is derived from Deep-SVDD-PyTorch and its local time-series
extension. The retained MIT license is distributed with this repository.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DeepSVDDModelError(ValueError):
    """Raised when the requested network shape cannot be constructed."""


def _pooled_size(value: int) -> int:
    return value // 2 // 2 // 2


class TimeSeriesLeNet(nn.Module):
    """Three-stage, bias-free encoder matching the legacy TS network."""

    def __init__(
        self,
        *,
        window_size: int,
        feature_count: int,
        rep_dim: int = 128,
    ) -> None:
        super().__init__()
        pooled_time = _pooled_size(window_size)
        pooled_features = _pooled_size(feature_count)
        if pooled_time < 1 or pooled_features < 1:
            raise DeepSVDDModelError(
                "window_size and feature_count must each be at least 8"
            )
        self.rep_dim = int(rep_dim)
        self.window_size = int(window_size)
        self.feature_count = int(feature_count)
        self.pooled_time = pooled_time
        self.pooled_features = pooled_features
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=5, bias=False, padding=2
        )
        self.bn1 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=5, bias=False, padding=2
        )
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=5, bias=False, padding=2
        )
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        flat_size = 128 * pooled_time * pooled_features
        self.fc1 = nn.Linear(flat_size, self.rep_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.pool(F.leaky_relu(self.bn1(self.conv1(values))))
        values = self.pool(F.leaky_relu(self.bn2(self.conv2(values))))
        values = self.pool(F.leaky_relu(self.bn3(self.conv3(values))))
        return self.fc1(values.reshape(values.shape[0], -1))


class TimeSeriesLeNetAutoencoder(nn.Module):
    """Autoencoder whose encoder keys transfer directly to TimeSeriesLeNet."""

    def __init__(
        self,
        *,
        window_size: int,
        feature_count: int,
        rep_dim: int = 128,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.feature_count = int(feature_count)
        self.rep_dim = int(rep_dim)
        self.h1 = window_size // 2
        self.w1 = feature_count // 2
        self.h2 = self.h1 // 2
        self.w2 = self.w1 // 2
        self.h3 = self.h2 // 2
        self.w3 = self.w2 // 2
        if self.h3 < 1 or self.w3 < 1:
            raise DeepSVDDModelError(
                "window_size and feature_count must each be at least 8"
            )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.conv1.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )
        self.bn1 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.conv2.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.conv3.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        flat_size = 128 * self.h3 * self.w3
        self.fc1 = nn.Linear(flat_size, rep_dim, bias=False)
        self.bn_fc = nn.BatchNorm1d(
            rep_dim, eps=1e-4, affine=False
        )
        self.dec_fc = nn.Linear(rep_dim, flat_size, bias=False)
        nn.init.xavier_uniform_(
            self.dec_fc.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv1.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )
        self.bn_d1 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv2.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )
        self.bn_d2 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.deconv3 = nn.ConvTranspose2d(
            32, 1, kernel_size=5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv3.weight,
            gain=nn.init.calculate_gain("leaky_relu"),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.pool(F.leaky_relu(self.bn1(self.conv1(values))))
        values = self.pool(F.leaky_relu(self.bn2(self.conv2(values))))
        values = self.pool(F.leaky_relu(self.bn3(self.conv3(values))))
        values = values.reshape(values.shape[0], -1)
        values = self.bn_fc(self.fc1(values))
        values = F.leaky_relu(self.dec_fc(values))
        values = values.reshape(values.shape[0], 128, self.h3, self.w3)
        values = self.deconv1(values)
        values = F.interpolate(
            F.leaky_relu(self.bn_d1(values)),
            size=(self.h2, self.w2),
        )
        values = self.deconv2(values)
        values = F.interpolate(
            F.leaky_relu(self.bn_d2(values)),
            size=(self.h1, self.w1),
        )
        values = self.deconv3(values)
        return F.interpolate(
            torch.sigmoid(values),
            size=(self.window_size, self.feature_count),
        )


def initialize_from_autoencoder(
    encoder: TimeSeriesLeNet,
    autoencoder: TimeSeriesLeNetAutoencoder,
) -> None:
    """Copy matching encoder tensors exactly as in the legacy wrapper."""

    encoder_state = encoder.state_dict()
    matching_state = {
        key: value
        for key, value in autoencoder.state_dict().items()
        if key in encoder_state
    }
    encoder_state.update(matching_state)
    encoder.load_state_dict(encoder_state)
