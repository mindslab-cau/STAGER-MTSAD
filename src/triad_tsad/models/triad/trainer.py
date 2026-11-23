"""Device-aware training and scoring for TriAD."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def _model_output(
    value: torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None] | None]:
    if isinstance(value, tuple):
        return value
    return value, None


class TriADTrainer:
    """Optimize the next-step MSE and optional representation losses."""

    def __init__(self, device: str | torch.device | None = None):
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.logger = logging.getLogger(__name__)
        self._warned_missing_embedding = False
        self._warned_missing_classifier = False

    def _build_optimizer(
        self,
        model: nn.Module,
        optimizer_class: type[torch.optim.Optimizer],
        lr: float,
        lr_global: float | None,
        lr_aux_classifier: float | None,
    ) -> torch.optim.Optimizer:
        embedding_params: list[nn.Parameter] = []
        if getattr(model, "global_embedding", None) is not None:
            embedding_params = list(model.global_embedding.parameters())

        classifier_params: list[nn.Parameter] = []
        if getattr(model, "global_classifier", None) is not None:
            classifier_params = list(model.global_classifier.parameters())

        special_ids = {
            id(parameter)
            for parameter in embedding_params + classifier_params
        }
        model_params = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in special_ids
        ]

        groups: list[dict[str, Any]] = []
        if model_params:
            groups.append({"params": model_params, "lr": lr})
        if embedding_params:
            groups.append(
                {
                    "params": embedding_params,
                    "lr": lr * 0.1 if lr_global is None else lr_global,
                }
            )
        if classifier_params:
            groups.append(
                {
                    "params": classifier_params,
                    "lr": lr if lr_aux_classifier is None else lr_aux_classifier,
                }
            )
        if not groups:
            raise ValueError("No parameters were found for optimization.")
        return optimizer_class(groups)

    def _global_embeddings(
        self,
        model: nn.Module,
        *,
        normalize: bool,
        eps: float,
    ) -> torch.Tensor | None:
        embedding = getattr(model, "global_embedding", None)
        if embedding is None:
            if not self._warned_missing_embedding:
                self.logger.warning(
                    "Global embeddings are unavailable; matching auxiliary "
                    "losses are skipped."
                )
                self._warned_missing_embedding = True
            return None

        if hasattr(embedding, "get_aux_profiles"):
            value = embedding.get_aux_profiles()
            if value is not None:
                return F.normalize(value, dim=-1, eps=eps) if normalize else value

        if hasattr(model, "get_global_embeddings"):
            return model.get_global_embeddings(normalize=normalize, eps=eps)

        value = embedding.get_for_forward(
            training=model.training,
            use_ema_in_forward=getattr(model, "use_ema_in_forward", False),
        )
        return F.normalize(value, dim=-1, eps=eps) if normalize else value

    def _global_logits(
        self,
        model: nn.Module,
        target: torch.Tensor | None,
        *,
        normalize: bool,
        eps: float,
    ) -> torch.Tensor | None:
        classifier = getattr(model, "global_classifier", None)
        if classifier is None:
            if not self._warned_missing_classifier:
                self.logger.warning(
                    "The global classifier is unavailable; its auxiliary "
                    "loss is skipped."
                )
                self._warned_missing_classifier = True
            return None

        if hasattr(model, "get_global_embedding_logits"):
            return model.get_global_embedding_logits(
                target,
                normalize=normalize,
                eps=eps,
            )

        value = target
        if value is None:
            value = self._global_embeddings(model, normalize=normalize, eps=eps)
        elif normalize:
            value = F.normalize(value, dim=-1, eps=eps)
        return None if value is None else classifier(value)

    def _auxiliary_losses(
        self,
        model: nn.Module,
        intermediates: dict[str, torch.Tensor | None] | None,
        *,
        euclid_weight: float,
        cosine_weight: float,
        orthogonal_weight: float,
        classifier_weight: float,
        norm_eps: float,
        classifier_use_norm: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        values = intermediates or {}
        global_value = values.get("global")
        conditioning_value = values.get("conditioning")

        normalized_global = (
            global_value
            if global_value is not None
            else self._global_embeddings(model, normalize=True, eps=norm_eps)
        )
        target_name = getattr(model, "aux_target", "global")
        if target_name == "global":
            target = normalized_global
        elif target_name == "conditioning":
            target = conditioning_value
        else:
            raise ValueError(
                "aux_target must be either 'global' or 'conditioning'."
            )

        losses: dict[str, torch.Tensor] = {}
        if (
            target is not None
            and target.size(0) >= 2
            and (euclid_weight > 0 or cosine_weight > 0 or orthogonal_weight > 0)
        ):
            count = target.size(0)
            off_diagonal = ~torch.eye(
                count,
                dtype=torch.bool,
                device=target.device,
            )
            cosine = target @ target.t()
            if euclid_weight > 0:
                distances = torch.cdist(target, target, p=2)
                losses["euclid"] = 2.0 - distances[off_diagonal].mean()
            if cosine_weight > 0:
                losses["cosine"] = (cosine[off_diagonal] + 1.0).mean()
            if orthogonal_weight > 0:
                identity = torch.eye(
                    count,
                    dtype=target.dtype,
                    device=target.device,
                )
                losses["orthogonal"] = ((cosine - identity) ** 2).mean()

        if classifier_weight > 0:
            logits = self._global_logits(
                model,
                target,
                normalize=classifier_use_norm,
                eps=norm_eps,
            )
            if logits is not None:
                labels = torch.arange(logits.size(0), device=logits.device)
                losses["classifier"] = F.cross_entropy(logits, labels)

        if not losses:
            return losses, None

        total = torch.zeros((), device=next(model.parameters()).device)
        weights = {
            "euclid": euclid_weight,
            "cosine": cosine_weight,
            "orthogonal": orthogonal_weight,
            "classifier": classifier_weight,
        }
        for name, value in losses.items():
            total = total + weights[name] * value
        return losses, total

    def fit(
        self,
        train_loader,
        model: nn.Module,
        *,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 1e-3,
        scheduler=None,
        epochs: int = 100,
        valid_loader=None,
        early_stopping_patience: int = 10,
        lr_global: float | None = None,
        lr_aux_classifier: float | None = None,
        ema_alpha: float = 1e-3,
        aux_euclid_weight: float = 0.0,
        aux_cos_weight: float = 0.0,
        aux_ortho_weight: float = 0.0,
        aux_cls_weight: float = 0.0,
        aux_norm_eps: float = 1e-8,
        aux_cls_use_norm: bool = False,
        best_state_policy: str = "snapshot",
        epoch_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Fit in memory and restore the lowest-loss state.

        The callback is the only training-event hook. This method does not
        create files or contact an external service.
        """

        if epochs < 1:
            raise ValueError("epochs must be at least one.")
        if len(train_loader) == 0:
            raise ValueError("train_loader must contain at least one batch.")
        if best_state_policy not in {"reference", "snapshot"}:
            raise ValueError(
                "best_state_policy must be 'reference' or 'snapshot'."
            )

        model.to(self.device)
        loss_function = nn.MSELoss()
        optimizer = self._build_optimizer(
            model,
            optimizer_class,
            lr,
            lr_global,
            lr_aux_classifier,
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []

        epoch_iterator = tqdm(
            range(epochs),
            total=epochs,
            desc="TriAD training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                given = batch["given"].to(self.device)
                answer = batch["answer"].to(self.device)
                prediction, intermediates = _model_output(model(given))
                loss = loss_function(prediction, answer)

                _, auxiliary = self._auxiliary_losses(
                    model,
                    intermediates,
                    euclid_weight=aux_euclid_weight,
                    cosine_weight=aux_cos_weight,
                    orthogonal_weight=aux_ortho_weight,
                    classifier_weight=aux_cls_weight,
                    norm_eps=aux_norm_eps,
                    classifier_use_norm=aux_cls_use_norm,
                )
                if auxiliary is not None:
                    loss = loss + auxiliary

                loss.backward()
                optimizer.step()
                embedding = getattr(model, "global_embedding", None)
                if hasattr(embedding, "ema_update"):
                    with torch.no_grad():
                        embedding.ema_update(ema_alpha)
                total_loss += float(loss.detach())

            average_train_loss = total_loss / len(train_loader)
            train_losses.append(average_train_loss)

            monitored_loss = average_train_loss
            payload = {
                "epoch": float(epoch + 1),
                "train_loss": average_train_loss,
            }
            if valid_loader is not None:
                monitored_loss = self.evaluate(valid_loader, model)
                validation_losses.append(monitored_loss)
                payload["validation_loss"] = monitored_loss
            if epoch_callback is not None:
                epoch_callback(payload)

            should_stop = False
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_epoch = epoch + 1
                state = model.state_dict()
                best_state = (
                    copy.deepcopy(state)
                    if best_state_policy == "snapshot"
                    else state
                )
                patience = 0
            else:
                patience += 1
                should_stop = patience >= early_stopping_patience

            postfix: dict[str, str] = {
                "train": f"{average_train_loss:.6f}",
                "best": f"{best_loss:.6f}",
                "patience": f"{patience}/{early_stopping_patience}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if valid_loader is not None:
                postfix["valid"] = f"{monitored_loss:.6f}"
            epoch_iterator.set_postfix(postfix)

            if should_stop:
                break

            if scheduler is not None:
                scheduler.step()

        if best_state is not None:
            model.load_state_dict(best_state)

        history: dict[str, Any] = {
            "train_loss": train_losses,
            "validation_loss": validation_losses or None,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_state_policy": best_state_policy,
        }
        return model, history

    def evaluate(self, data_loader, model: nn.Module) -> float:
        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch.")
        model.eval()
        loss_function = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                given = batch["given"].to(self.device)
                answer = batch["answer"].to(self.device)
                prediction, _ = _model_output(model(given))
                total_loss += float(loss_function(prediction, answer))
        return total_loss / len(data_loader)

    def score(
        self,
        data_loader,
        model: nn.Module,
    ) -> tuple[float, np.ndarray]:
        """Return average MSE and squared errors shaped ``[time, feature]``."""

        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch.")
        model.eval()
        feature_scores: list[np.ndarray] = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                given = batch["given"].to(self.device)
                answer = batch["answer"].to(self.device)
                prediction, _ = _model_output(model(given))
                squared_error = (prediction - answer).square()
                feature_scores.append(squared_error.cpu().numpy())
                total_loss += float(squared_error.mean())

        return total_loss / len(data_loader), np.concatenate(feature_scores, axis=0)

    def predict(self, data_loader, model: nn.Module) -> np.ndarray:
        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch.")
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for batch in data_loader:
                given = batch["given"].to(self.device)
                prediction, _ = _model_output(model(given))
                predictions.append(prediction.cpu().numpy())
        return np.concatenate(predictions, axis=0)


__all__ = ["TriADTrainer"]
