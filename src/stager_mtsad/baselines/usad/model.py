"""USAD autoencoder adapted for portable offline anomaly scoring."""

from __future__ import annotations

import torch
from torch import nn


class USADModelError(ValueError):
    """Raised when a tensor or model shape is incompatible with USAD."""


class Encoder(nn.Module):
    """Three-layer encoder used by the published USAD implementation."""

    def __init__(self, input_size: int, latent_dim: int) -> None:
        super().__init__()
        if input_size < 4:
            raise USADModelError("input_size must be at least 4")
        if latent_dim < 1:
            raise USADModelError("latent_dim must be positive")
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.linear3 = nn.Linear(input_size // 4, latent_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.linear1(inputs))
        hidden = self.activation(self.linear2(hidden))
        return self.activation(self.linear3(hidden))


class Decoder(nn.Module):
    """Sigmoid decoder used by the published USAD implementation."""

    def __init__(self, latent_dim: int, output_size: int) -> None:
        super().__init__()
        if output_size < 4:
            raise USADModelError("output_size must be at least 4")
        if latent_dim < 1:
            raise USADModelError("latent_dim must be positive")
        self.linear1 = nn.Linear(latent_dim, output_size // 4)
        self.linear2 = nn.Linear(output_size // 4, output_size // 2)
        self.linear3 = nn.Linear(output_size // 2, output_size)
        self.activation = nn.ReLU(inplace=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.linear1(latent))
        hidden = self.activation(self.linear2(hidden))
        return self.output_activation(self.linear3(hidden))


class USAD(nn.Module):
    """Two-decoder USAD model accepting flat or ``[batch, window, feature]`` input."""

    def __init__(
        self,
        *,
        num_channels: int,
        window_size: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise USADModelError("num_channels must be positive")
        if window_size < 1:
            raise USADModelError("window_size must be positive")
        self.num_channels = int(num_channels)
        self.window_size = int(window_size)
        self.input_size = self.num_channels * self.window_size
        self.latent_dim = int(latent_dim)
        self.encoder = Encoder(self.input_size, self.latent_dim)
        self.decoder1 = Decoder(self.latent_dim, self.input_size)
        self.decoder2 = Decoder(self.latent_dim, self.input_size)

    def _flatten(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        if inputs.ndim == 2:
            if inputs.shape[1] != self.input_size:
                raise USADModelError(
                    f"expected flat input width {self.input_size}, "
                    f"received {inputs.shape[1]}"
                )
            return inputs, inputs.shape
        if inputs.ndim != 3:
            raise USADModelError(
                "input must have shape [batch, flattened] or "
                "[batch, window, feature]"
            )
        if inputs.shape[1:] != (self.window_size, self.num_channels):
            raise USADModelError(
                "expected window shape "
                f"[{self.window_size}, {self.num_channels}], received "
                f"{list(inputs.shape[1:])}"
            )
        return inputs.reshape(inputs.shape[0], -1), inputs.shape

    @staticmethod
    def _restore(
        outputs: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        if len(original_shape) == 3:
            return outputs.reshape(original_shape)
        return outputs

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct input through each decoder."""

        flattened, original_shape = self._flatten(inputs)
        latent = self.encoder(flattened)
        reconstruction1 = self.decoder1(latent)
        reconstruction2 = self.decoder2(latent)
        return (
            self._restore(reconstruction1, original_shape),
            self._restore(reconstruction2, original_shape),
        )

    def losses(
        self,
        inputs: torch.Tensor,
        *,
        epoch: int,
        warmup_epochs: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two historical training losses for a one-based epoch."""

        if epoch < 1:
            raise USADModelError("epoch must be one-based and positive")
        if warmup_epochs < 0:
            raise USADModelError("warmup_epochs must be non-negative")
        flattened, _ = self._flatten(inputs)
        latent = self.encoder(flattened)
        reconstruction1 = self.decoder1(latent)
        reconstruction2 = self.decoder2(latent)
        adversarial_reconstruction = self.decoder2(
            self.encoder(reconstruction1)
        )

        first_error = torch.mean(
            torch.square(flattened - reconstruction1)
        )
        second_error = torch.mean(
            torch.square(flattened - reconstruction2)
        )
        adversarial_error = torch.mean(
            torch.square(flattened - adversarial_reconstruction)
        )
        if epoch <= warmup_epochs:
            return first_error, second_error

        adversarial_epoch = epoch - warmup_epochs
        weight = 1.0 / adversarial_epoch
        loss1 = weight * first_error + (1.0 - weight) * adversarial_error
        loss2 = weight * second_error - (1.0 - weight) * adversarial_error
        return loss1, loss2

    def legacy_feature_scores(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the project-local score definition as ``[batch, feature]``.

        The local experiment trainer used only decoder 1 and averaged squared
        reconstruction errors over the complete input window. Its ``alpha``
        and ``beta`` inference arguments were unused. This method intentionally
        retains that behavior.
        """

        if inputs.ndim != 3:
            raise USADModelError(
                "legacy feature scores require [batch, window, feature] input"
            )
        reconstruction1, _ = self(inputs)
        return torch.mean(torch.square(inputs - reconstruction1), dim=1)

    def combined_scores(
        self,
        inputs: torch.Tensor,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> torch.Tensor:
        """Return the upstream paper-style flattened score.

        This is provided for explicit comparison only. The native runner uses
        :meth:`legacy_feature_scores` to preserve the project-local protocol.
        """

        flattened, _ = self._flatten(inputs)
        reconstruction1 = self.decoder1(self.encoder(flattened))
        reconstruction2 = self.decoder2(self.encoder(reconstruction1))
        return alpha * torch.mean(
            torch.square(flattened - reconstruction1), dim=1
        ) + beta * torch.mean(
            torch.square(flattened - reconstruction2), dim=1
        )
