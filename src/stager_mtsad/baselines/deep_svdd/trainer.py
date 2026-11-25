"""Deep SVDD objective training with the migrated time-series network.

The objective follows the retained MIT-licensed Deep-SVDD-PyTorch source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from stager_mtsad.utils import set_random_seed

from .config import DeepSVDDConfig
from .model import (
    TimeSeriesLeNet,
    TimeSeriesLeNetAutoencoder,
    initialize_from_autoencoder,
)


class DeepSVDDTrainingError(RuntimeError):
    """Raised when a Deep SVDD training phase cannot be completed."""


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Losses and learned hypersphere values for one fitted run."""

    autoencoder_loss: tuple[float, ...]
    objective_loss: tuple[float, ...]
    radius: float
    center: np.ndarray


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``auto`` and reject unavailable accelerator requests."""

    if str(device) == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise DeepSVDDTrainingError(
            "CUDA was requested but is not available"
        )
    return resolved


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def _optimizer(
    parameters: Any,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    normalized = name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=normalized == "amsgrad",
    )


def _mean_epoch_loss(
    losses: list[float],
    phase: str,
) -> float:
    if not losses:
        raise DeepSVDDTrainingError(
            f"{phase} produced no training batches; reduce batch_size"
        )
    return float(np.mean(losses))


class DeepSVDDTrainer:
    """Fit pretraining and Deep SVDD objectives without external side effects."""

    def __init__(
        self,
        config: DeepSVDDConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.model: TimeSeriesLeNet | None = None
        self.autoencoder: TimeSeriesLeNetAutoencoder | None = None
        self.history: TrainingHistory | None = None

    def _pretrain(
        self,
        dataset: Dataset[Any],
        *,
        window_size: int,
        feature_count: int,
    ) -> tuple[TimeSeriesLeNetAutoencoder, tuple[float, ...]]:
        autoencoder = TimeSeriesLeNetAutoencoder(
            window_size=window_size,
            feature_count=feature_count,
            rep_dim=self.config.rep_dim,
        ).to(self.device)
        optimizer = _optimizer(
            autoencoder.parameters(),
            name=self.config.optimizer,
            learning_rate=self.config.autoencoder_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.config.learning_rate_milestones),
            gamma=0.1,
        )
        loader = _loader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        epoch_losses: list[float] = []
        autoencoder.train()
        epoch_iterator = tqdm(
            range(self.config.autoencoder_num_epochs),
            total=self.config.autoencoder_num_epochs,
            desc="DeepSVDD pretraining",
            unit="epoch",
            dynamic_ncols=True,
        )
        for _ in epoch_iterator:
            batch_losses: list[float] = []
            for batch in loader:
                values = batch["input"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = autoencoder(values)
                sample_error = torch.sum(
                    torch.square(reconstruction - values),
                    dim=tuple(range(1, reconstruction.ndim)),
                )
                loss = torch.mean(sample_error)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            epoch_loss = _mean_epoch_loss(
                batch_losses, "autoencoder pretraining"
            )
            epoch_losses.append(epoch_loss)
            epoch_iterator.set_postfix(
                loss=f"{epoch_loss:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            scheduler.step()
        return autoencoder, tuple(epoch_losses)

    def _initialize_center(
        self,
        loader: DataLoader[Any],
        model: TimeSeriesLeNet,
        *,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        center = torch.zeros(
            model.rep_dim,
            device=self.device,
        )
        sample_count = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                values = batch["input"].to(self.device)
                outputs = model(values)
                sample_count += outputs.shape[0]
                center += torch.sum(outputs, dim=0)
        if sample_count == 0:
            raise DeepSVDDTrainingError(
                "center initialization produced no batches"
            )
        center /= sample_count
        center[(torch.abs(center) < epsilon) & (center < 0)] = -epsilon
        center[(torch.abs(center) < epsilon) & (center > 0)] = epsilon
        return center

    def fit(
        self,
        dataset: Dataset[Any],
        *,
        window_size: int,
        feature_count: int,
    ) -> DeepSVDDTrainer:
        """Fit the historical autoencoder and hypersphere objectives."""

        set_random_seed(self.config.seed)
        model = TimeSeriesLeNet(
            window_size=window_size,
            feature_count=feature_count,
            rep_dim=self.config.rep_dim,
        ).to(self.device)
        autoencoder_losses: tuple[float, ...] = ()
        if self.config.pretrain:
            autoencoder, autoencoder_losses = self._pretrain(
                dataset,
                window_size=window_size,
                feature_count=feature_count,
            )
            initialize_from_autoencoder(model, autoencoder)
            self.autoencoder = autoencoder

        loader = _loader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        center = self._initialize_center(loader, model)
        radius = torch.tensor(0.0, device=self.device)
        optimizer = _optimizer(
            model.parameters(),
            name=self.config.optimizer,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.config.learning_rate_milestones),
            gamma=0.1,
        )
        epoch_losses: list[float] = []
        model.train()
        epoch_iterator = tqdm(
            range(self.config.num_epochs),
            total=self.config.num_epochs,
            desc="DeepSVDD training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            batch_losses: list[float] = []
            for batch in loader:
                values = batch["input"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(values)
                distance = torch.sum(
                    torch.square(outputs - center), dim=1
                )
                if self.config.objective == "soft-boundary":
                    scores = distance - torch.square(radius)
                    loss = torch.square(radius) + (
                        1.0 / self.config.nu
                    ) * torch.mean(
                        torch.maximum(
                            torch.zeros_like(scores), scores
                        )
                    )
                else:
                    loss = torch.mean(distance)
                loss.backward()
                optimizer.step()
                if (
                    self.config.objective == "soft-boundary"
                    and epoch >= self.config.warm_up_epochs
                ):
                    quantile = np.quantile(
                        np.sqrt(
                            distance.detach().cpu().numpy()
                        ),
                        1.0 - self.config.nu,
                    )
                    radius.data = torch.tensor(
                        quantile,
                        device=self.device,
                    )
                batch_losses.append(float(loss.detach().cpu()))
            epoch_loss = _mean_epoch_loss(
                batch_losses, "Deep SVDD training"
            )
            epoch_losses.append(epoch_loss)
            epoch_iterator.set_postfix(
                loss=f"{epoch_loss:.6f}",
                radius=f"{float(radius.detach().cpu()):.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            scheduler.step()

        self.model = model
        self.history = TrainingHistory(
            autoencoder_loss=autoencoder_losses,
            objective_loss=tuple(epoch_losses),
            radius=float(radius.detach().cpu()),
            center=center.detach().cpu().numpy().copy(),
        )
        return self

    def score(
        self,
        dataset: Dataset[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical anomaly scores and their raw row anchors."""

        if self.model is None or self.history is None:
            raise DeepSVDDTrainingError(
                "fit must be called before score"
            )
        loader = _loader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        center = torch.as_tensor(
            self.history.center,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        radius = torch.tensor(
            self.history.radius,
            device=self.device,
            dtype=center.dtype,
        )
        scores: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                values = batch["input"].to(self.device)
                outputs = self.model(values)
                distance = torch.sum(
                    torch.square(outputs - center), dim=1
                )
                if self.config.objective == "soft-boundary":
                    anomaly_score = distance - torch.square(radius)
                else:
                    anomaly_score = distance
                scores.append(anomaly_score.cpu().numpy())
                anchors.append(batch["anchor"].cpu().numpy())
        if not scores:
            raise DeepSVDDTrainingError(
                "test split contains no scoreable windows"
            )
        return (
            np.concatenate(scores).astype(np.float64, copy=False),
            np.concatenate(anchors).astype(np.int64, copy=False),
        )
