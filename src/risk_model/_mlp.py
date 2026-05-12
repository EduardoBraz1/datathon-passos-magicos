"""PyTorch MLP with focal-loss training, sklearn-compatible wrapper."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

from .config import MLPConfig, MLP_CONFIG, RANDOM_STATE

log = logging.getLogger(__name__)


def _autodevice() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _MLP(nn.Module):
    def __init__(self, in_features: int, cfg: MLPConfig):
        super().__init__()
        layers = []
        prev = in_features
        for h in cfg.hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss with optional alpha (positive-class weight)."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn-compatible wrapper around the PyTorch MLP."""

    def __init__(self, cfg: Optional[MLPConfig] = None, device: Optional[str] = None,
                 random_state: int = RANDOM_STATE):
        self.cfg = cfg or MLP_CONFIG
        self.device = device
        self.random_state = random_state

    def _build_optim(self, model, steps_per_epoch):
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.cfg.lr,
            steps_per_epoch=max(1, steps_per_epoch),
            epochs=self.cfg.epochs,
            pct_start=self.cfg.one_cycle_pct_start,
        )
        return opt, sched

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = self.device or _autodevice()
        self.device_ = device

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        in_features = X.shape[1]
        self.in_features_ = in_features
        prevalence = float(y.mean())
        alpha = float(1.0 - prevalence)  # weight on the positive class

        model = _MLP(in_features, self.cfg).to(device)
        loss_fn = FocalLoss(alpha=alpha, gamma=self.cfg.focal_gamma).to(device)

        ds = TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        )
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(device)

        opt, sched = self._build_optim(model, steps_per_epoch=len(loader))

        best_state = None
        best_score = math.inf
        epochs_since_best = 0
        history = []
        for epoch in range(self.cfg.epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                if has_val:
                    val_logits = model(Xv)
                    val_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, yv).item()
                else:
                    val_loss = float("nan")
            history.append(val_loss)
            if has_val and val_loss < best_score - 1e-5:
                best_score = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            if has_val and epochs_since_best >= self.cfg.patience:
                log.info("MLP early-stopping at epoch %d (best val loss %.4f)", epoch + 1, best_score)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.classes_ = np.array([0, 1])
        self.history_ = history
        self.prevalence_ = prevalence
        return self

    def _logits(self, X) -> np.ndarray:
        self.model_.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(X).to(self.device_)
            out = self.model_(t).cpu().numpy()
        return out

    def decision_function(self, X) -> np.ndarray:
        return self._logits(X)

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self._logits(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True):
        return {
            "cfg": self.cfg,
            "device": self.device,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def to_state_dict(self):
        return {
            "model_state": {k: v.cpu() for k, v in self.model_.state_dict().items()},
            "in_features": self.in_features_,
            "cfg": asdict(self.cfg),
            "prevalence": self.prevalence_,
        }

    @classmethod
    def from_state_dict(cls, state, cfg: Optional[MLPConfig] = None):
        cfg = cfg or MLPConfig(**state["cfg"])
        obj = cls(cfg=cfg)
        obj.device_ = _autodevice()
        model = _MLP(state["in_features"], cfg).to(obj.device_)
        model.load_state_dict({k: v.to(obj.device_) for k, v in state["model_state"].items()})
        obj.model_ = model
        obj.in_features_ = state["in_features"]
        obj.prevalence_ = state["prevalence"]
        obj.classes_ = np.array([0, 1])
        return obj
