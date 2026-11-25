"""Offline trainer and native RunResult adapter for Anomaly Transformer."""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ...contracts import RunResult
from ...utils import set_random_seed
from .model import AnomalyTransformer

LOGGER = logging.getLogger(__name__)


def _batch_input(batch: Any, device: torch.device) -> torch.Tensor:
    if isinstance(batch, Mapping):
        if "given" not in batch:
            raise ValueError("mapping batches must contain a 'given' tensor")
        value = batch["given"]
    elif isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("sequence batches must not be empty")
        value = batch[0]
    else:
        value = batch

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 3:
        raise ValueError("input batches must have shape [batch, window, feature]")
    return tensor


def _batch_metadata(batch: Any, names: tuple[str, ...]) -> np.ndarray | None:
    if not isinstance(batch, Mapping):
        return None
    for name in names:
        if name not in batch:
            continue
        value = batch[name]
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        return array.reshape(-1)
    return None


def _combined_metadata(
    explicit: Any | None,
    collected: list[np.ndarray],
) -> Any | None:
    if explicit is not None:
        return explicit
    if not collected:
        return None
    return np.concatenate(collected)


class AnomalyTransformerTrainer:
    """Preserve the local minimax trainer without tracking dependencies."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        association_weight: float = 0.1,
        temperature: float = 50.0,
    ) -> None:
        if association_weight < 0:
            raise ValueError("association_weight must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.association_weight = float(association_weight)
        self.temperature = float(temperature)

    @staticmethod
    def kl_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Legacy association KL reduction."""

        result = first * (
            torch.log(first + 0.0001) - torch.log(second + 0.0001)
        )
        return torch.mean(torch.sum(result, dim=-1), dim=1)

    @staticmethod
    def _normalized_prior(
        prior: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        denominator = torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1)
        return prior / denominator.repeat(1, 1, 1, window_size)

    def _association_training_losses(
        self,
        series: list[torch.Tensor],
        prior: list[torch.Tensor],
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prior:
            raise ValueError("output_attention=True must produce association priors")

        series_loss: torch.Tensor | float = 0.0
        prior_loss: torch.Tensor | float = 0.0
        for series_layer, prior_layer in zip(series, prior, strict=True):
            normalized_prior = self._normalized_prior(prior_layer, window_size)
            detached_prior = normalized_prior.detach()
            series_loss = series_loss + (
                torch.mean(self.kl_loss(series_layer, detached_prior))
                + torch.mean(self.kl_loss(detached_prior, series_layer))
            )
            prior_loss = prior_loss + (
                torch.mean(self.kl_loss(normalized_prior, series_layer.detach()))
                + torch.mean(self.kl_loss(series_layer.detach(), normalized_prior))
            )

        return series_loss / len(prior), prior_loss / len(prior)

    @staticmethod
    def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
        """Freeze the selected epoch instead of aliasing live parameters."""

        return copy.deepcopy(model.state_dict())

    @staticmethod
    def _adjust_learning_rate(
        optimizer: torch.optim.Optimizer,
        epoch: int,
        initial_learning_rate: float,
    ) -> None:
        if optimizer.__class__.__name__ in {"Adam", "AdamW", "Adamax"}:
            return
        if epoch % 10 == 0 and epoch > 0:
            new_learning_rate = initial_learning_rate * (0.9 ** (epoch // 10))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = new_learning_rate

    def validate(
        self,
        data_loader: Iterable[Any],
        model: AnomalyTransformer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        model.eval()
        loss_1: list[float] = []
        loss_2: list[float] = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = _batch_input(batch, self.device)
                output, series, prior, _ = model(inputs)
                series_loss, prior_loss = self._association_training_losses(
                    series,
                    prior,
                    inputs.shape[1],
                )
                reconstruction_loss = criterion(output, inputs)
                loss_1.append(
                    (
                        reconstruction_loss
                        - self.association_weight * series_loss
                    ).item()
                )
                loss_2.append(
                    (
                        reconstruction_loss
                        + self.association_weight * prior_loss
                    ).item()
                )

        if not loss_1:
            raise ValueError("validation loader must contain at least one batch")
        return float(np.average(loss_1)), float(np.average(loss_2))

    def train(
        self,
        train_loader: Iterable[Any],
        model: AnomalyTransformer,
        *,
        num_epochs: int = 100,
        valid_loader: Iterable[Any] | None = None,
        early_stopping_patience: int = 10,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 0.001,
    ) -> tuple[AnomalyTransformer, dict[str, list[float] | None]]:
        """Train with the adapted source's two-backward minimax update."""

        if num_epochs < 1:
            raise ValueError("num_epochs must be at least one")
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least one")

        model.to(self.device)
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_model: dict[str, torch.Tensor] | None = None
        patience_counter = 0
        train_losses: list[float] = []
        validation_losses_1: list[float] = []
        validation_losses_2: list[float] = []

        epoch_iterator = tqdm(
            range(num_epochs),
            total=num_epochs,
            desc="AnomalyTransformer training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                inputs = _batch_input(batch, self.device)
                optimizer.zero_grad()
                output, series, prior, _ = model(inputs)
                series_loss, prior_loss = self._association_training_losses(
                    series,
                    prior,
                    inputs.shape[1],
                )
                reconstruction_loss = criterion(output, inputs)
                loss_1 = (
                    reconstruction_loss
                    - self.association_weight * series_loss
                )
                loss_2 = (
                    reconstruction_loss
                    + self.association_weight * prior_loss
                )
                loss_1.backward(retain_graph=True)
                loss_2.backward()
                optimizer.step()

                epoch_loss += loss_1.item()
                batch_count += 1

            if batch_count == 0:
                raise ValueError("training loader must contain at least one batch")

            average_train_loss = epoch_loss / batch_count
            train_losses.append(average_train_loss)

            if valid_loader is not None:
                validation_loss_1, validation_loss_2 = self.validate(
                    valid_loader,
                    model,
                    criterion,
                )
                validation_losses_1.append(validation_loss_1)
                validation_losses_2.append(validation_loss_2)
                monitored_loss = validation_loss_1
            else:
                monitored_loss = average_train_loss

            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_model = self._state_snapshot(model)
                patience_counter = 0
            else:
                patience_counter += 1

            should_stop = patience_counter >= early_stopping_patience
            postfix = {
                "train": f"{average_train_loss:.6f}",
                "best": f"{best_loss:.6f}",
                "patience": (
                    f"{patience_counter}/{early_stopping_patience}"
                ),
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if valid_loader is not None:
                postfix["valid1"] = f"{validation_loss_1:.6f}"
                postfix["valid2"] = f"{validation_loss_2:.6f}"
            epoch_iterator.set_postfix(postfix)

            if should_stop:
                LOGGER.info("Early stopping at epoch %d", epoch + 1)
                if best_model is not None:
                    model.load_state_dict(best_model)
                break

            self._adjust_learning_rate(
                optimizer,
                epoch + 1,
                learning_rate,
            )

        if best_model is not None:
            model.load_state_dict(best_model)

        history: dict[str, list[float] | None] = {
            "train_loss": train_losses,
            "val_loss1": validation_losses_1 if valid_loader is not None else None,
            "val_loss2": validation_losses_2 if valid_loader is not None else None,
        }
        return model, history

    def _feature_score_batch(
        self,
        inputs: torch.Tensor,
        output: torch.Tensor,
        series: list[torch.Tensor],
        prior: list[torch.Tensor],
    ) -> torch.Tensor:
        squared_error = nn.functional.mse_loss(
            output,
            inputs,
            reduction="none",
        )
        series_discrepancy: torch.Tensor | None = None
        prior_discrepancy: torch.Tensor | None = None

        for series_layer, prior_layer in zip(series, prior, strict=True):
            normalized_prior = self._normalized_prior(
                prior_layer,
                inputs.shape[1],
            )
            current_series = (
                self.kl_loss(series_layer, normalized_prior.detach())
                * self.temperature
            )
            current_prior = (
                self.kl_loss(normalized_prior, series_layer.detach())
                * self.temperature
            )
            if series_discrepancy is None:
                series_discrepancy = current_series
                prior_discrepancy = current_prior
            else:
                series_discrepancy = series_discrepancy + current_series
                assert prior_discrepancy is not None
                prior_discrepancy = prior_discrepancy + current_prior

        if series_discrepancy is None or prior_discrepancy is None:
            raise ValueError("model did not return association tensors")

        temporal_weight = torch.softmax(
            -series_discrepancy - prior_discrepancy,
            dim=-1,
        ).unsqueeze(-1)
        # This is the local TSAD adaptation's score formula: retain a score per
        # feature and average the weighted reconstruction errors over time.
        return torch.mean(temporal_weight * squared_error, dim=1)

    def inference(
        self,
        data_loader: Iterable[Any],
        model: AnomalyTransformer,
        *,
        labels: Any | None = None,
        timestamps: Any | None = None,
        indices: Any | None = None,
        seed: int | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Return canonical scores while preserving the legacy score formula."""

        model.to(self.device)
        model.eval()
        feature_score_batches: list[np.ndarray] = []
        collected_labels: list[np.ndarray] = []
        collected_timestamps: list[np.ndarray] = []
        collected_indices: list[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = _batch_input(batch, self.device)
                output, series, prior, _ = model(inputs)
                feature_scores = self._feature_score_batch(
                    inputs,
                    output,
                    series,
                    prior,
                )
                feature_score_batches.append(
                    feature_scores.detach().cpu().numpy()
                )

                batch_labels = _batch_metadata(batch, ("label", "labels"))
                batch_timestamps = _batch_metadata(
                    batch,
                    ("timestamp", "timestamps", "ts"),
                )
                batch_indices = _batch_metadata(batch, ("index", "indices"))
                if batch_labels is not None:
                    collected_labels.append(batch_labels)
                if batch_timestamps is not None:
                    collected_timestamps.append(batch_timestamps)
                if batch_indices is not None:
                    collected_indices.append(batch_indices)

        if not feature_score_batches:
            raise ValueError("inference loader must contain at least one batch")

        feature_scores = np.concatenate(feature_score_batches, axis=0)
        diagnostic_point_scores = np.mean(feature_scores, axis=1)
        resolved_config = dict(config or {})
        resolved_config.setdefault(
            "association_weight",
            self.association_weight,
        )
        resolved_config.setdefault("temperature", self.temperature)
        resolved_metadata = {
            "baseline": "anomaly_transformer",
            "score_formula": (
                "mean_time(softmax(-series_kl-prior_kl)"
                "*squared_reconstruction_error)"
            ),
            "feature_score_orientation": "time_feature",
            "evaluation_score_orientation": "[feature, time]",
            "evaluation_score_source": "feature_scores_transposed",
            "alignment": "legacy_next_point",
            "output_attention": True,
            "diagnostic_global_mean": float(np.mean(feature_scores)),
            "point_score_status": "not_used_by_multistream_fpr_evaluation",
            "best_state_policy": "deep_state_dict_snapshot",
        }

        return RunResult(
            point_scores=diagnostic_point_scores,
            feature_scores=feature_scores,
            labels=_combined_metadata(labels, collected_labels),
            timestamps=_combined_metadata(timestamps, collected_timestamps),
            indices=_combined_metadata(indices, collected_indices),
            metadata=resolved_metadata,
            seed=seed,
            config=resolved_config,
        )


class AnomalyTransformerRunner:
    """Build, fit, and score a native Anomaly Transformer baseline."""

    def __init__(
        self,
        *,
        window_size: int,
        input_dim: int,
        output_dim: int | None = None,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = True,
        association_weight: float = 0.1,
        temperature: float = 50.0,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        output_dim = input_dim if output_dim is None else output_dim
        self.seed = set_random_seed(seed)
        self.model = AnomalyTransformer(
            window_size=window_size,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
        )
        self.trainer = AnomalyTransformerTrainer(
            device=device,
            association_weight=association_weight,
            temperature=temperature,
        )
        self.resolved_config = {
            "model": {
                "window_size": window_size,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "d_model": d_model,
                "n_heads": n_heads,
                "e_layers": e_layers,
                "d_ff": d_ff,
                "dropout": dropout,
                "activation": activation,
                "output_attention": output_attention,
            },
            "training": {
                "association_weight": association_weight,
                "seed": self.seed,
            },
            "scoring": {"temperature": temperature},
        }
        self.history: dict[str, list[float] | None] | None = None

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        num_epochs: int = 100,
        valid_loader: Iterable[Any] | None = None,
        early_stopping_patience: int = 10,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 0.001,
    ) -> dict[str, list[float] | None]:
        self.model, self.history = self.trainer.train(
            train_loader,
            self.model,
            num_epochs=num_epochs,
            valid_loader=valid_loader,
            early_stopping_patience=early_stopping_patience,
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
        )
        return self.history

    def score(
        self,
        data_loader: Iterable[Any],
        *,
        labels: Any | None = None,
        timestamps: Any | None = None,
        indices: Any | None = None,
        seed: int | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> RunResult:
        resolved_config = dict(self.resolved_config)
        if config is not None:
            resolved_config["run"] = dict(config)
        resolved_seed = self.seed if seed is None else seed
        if resolved_seed != self.seed:
            raise ValueError(
                "score seed must match the seed used to initialize the model"
            )
        result = self.trainer.inference(
            data_loader,
            self.model,
            labels=labels,
            timestamps=timestamps,
            indices=indices,
            seed=resolved_seed,
            config=resolved_config,
        )
        if self.history is not None:
            result.metadata["training_history"] = self.history
        return result

    def run(
        self,
        train_loader: Iterable[Any],
        test_loader: Iterable[Any],
        *,
        num_epochs: int = 100,
        valid_loader: Iterable[Any] | None = None,
        early_stopping_patience: int = 10,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 0.001,
        labels: Any | None = None,
        timestamps: Any | None = None,
        indices: Any | None = None,
        seed: int | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> RunResult:
        self.fit(
            train_loader,
            num_epochs=num_epochs,
            valid_loader=valid_loader,
            early_stopping_patience=early_stopping_patience,
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
        )
        return self.score(
            test_loader,
            labels=labels,
            timestamps=timestamps,
            indices=indices,
            seed=seed,
            config=config,
        )
