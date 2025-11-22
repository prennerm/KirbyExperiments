"""Gemeinsame Utilities für Training-Metriken."""
from __future__ import annotations

from typing import Dict, Optional


def build_training_metrics_snapshot(
    *,
    policy_loss: float,
    value_loss: float,
    entropy_loss: float,
    approx_kl: float,
    clip_fraction: float,
    total_loss: float,
    explained_variance: Optional[float] = None,
    extras: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Erzeugt ein konsistentes Dict mit Trainingsmetriken."""
    snapshot: Dict[str, float] = {
        "loss": {
            "total": total_loss,
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy_loss,
        },
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "loss_total": total_loss,
        "train/policy_loss": policy_loss,
        "train/value_loss": value_loss,
        "train/mc_loss": value_loss,
        "train/policy_gradient_loss": policy_loss,
        "train/entropy_loss": entropy_loss,
        "train/approx_kl": approx_kl,
        "train/clip_fraction": clip_fraction,
        "train/loss": total_loss,
    }
    if explained_variance is not None:
        snapshot["explained_variance"] = explained_variance
        snapshot["train/explained_variance"] = explained_variance
    if extras:
        snapshot.update(extras)
    return snapshot

