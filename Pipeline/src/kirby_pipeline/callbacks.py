"""Logging callbacks for the Kirby pipeline."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StatsWriter:
    def write_chunk(self, stats: List[Dict[str, Any]], final: bool = False) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Hook for writers that need explicit cleanup."""


class JsonStatsWriter(StatsWriter):
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def write_chunk(self, stats: List[Dict[str, Any]], final: bool = False) -> None:
        if not stats:
            return
        suffix = "final" if final else str(int(time.time()))
        file_path = self.base_path / f"stats_{suffix}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)


class CsvStatsWriter(StatsWriter):
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def write_chunk(self, stats: List[Dict[str, Any]], final: bool = False) -> None:
        if not stats:
            return
        flattened = [flatten_dict(entry) for entry in stats]
        fieldnames = sorted({key for row in flattened for key in row.keys()})
        suffix = "final" if final else str(int(time.time()))
        file_path = self.base_path / f"stats_{suffix}.csv"
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in flattened:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


class StatsCallback(BaseCallback):
    """Collects per-step stats from environments and writes them to disk."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        *,
        output_format: str = "csv",
        structured: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.structured = structured
        self.current_stats: List[Dict[str, Any]] = []
        self.writer = self._build_writer(output_format)

    def _build_writer(self, output_format: str) -> StatsWriter:
        fmt = output_format.lower()
        if fmt == "json":
            return JsonStatsWriter(self.save_path)
        if fmt == "csv":
            return CsvStatsWriter(self.save_path)
        raise ValueError(f"Unsupported stats output format: {output_format}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        stats_lists = self.training_env.get_attr("agent_stats")
        for env_idx, env_stats in enumerate(stats_lists):
            if not env_stats:
                continue
            for raw in env_stats:
                prepared = self._prepare_stat(raw, env_idx)
                self.current_stats.append(prepared)
        self.training_env.set_attr("agent_stats", [])
        if len(self.current_stats) >= self.save_freq:
            self._flush()

    def _on_training_end(self) -> None:
        self._flush(final=True)
        self.writer.close()

    def _prepare_stat(self, raw_stats: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
        normalised = _normalise_stat(raw_stats)
        normalised.setdefault("env_index", env_idx)
        normalised["total_steps"] = self.num_timesteps

        training_metrics: Optional[Dict[str, Any]] = None
        metrics_source = getattr(self.model, "latest_train_metrics", None)
        if isinstance(metrics_source, dict):
            training_metrics = _normalise_stat(metrics_source)
            if training_metrics:
                normalised["training"] = training_metrics

        if not self.structured:
            return normalised

        structured: Dict[str, Any] = {
            "step": normalised.get("step", 0),
            "total_steps": self.num_timesteps,
            "env": {
                "index": env_idx,
                "worker": normalised.get("worker", env_idx),
            },
            "reward": {
                "step": normalised.get("reward_step", 0.0),
                "total": normalised.get("reward_total", 0.0),
            },
            "progress": {
                "level": normalised.get("level_progress", 0),
                "delta": normalised.get("level_progress_delta", 0),
                "x": normalised.get("x_position", 0),
                "score": normalised.get("score", 0),
                "score_delta": normalised.get("score_delta", 0),
            },
            "boss": {
                "health": normalised.get("boss_health", 0),
                "delta": normalised.get("boss_health_delta", 0),
                "active": normalised.get("boss_active", False),
            },
            "status": {
                "health": normalised.get("health", 0),
                "lives": normalised.get("lives", 0),
                "game_state": normalised.get("game_state", 0),
                "warpstar": normalised.get("warpstar", False),
                "died": normalised.get("died", False),
            },
            "action": {
                "last": normalised.get("last_action", -1),
            },
        }
        if training_metrics:
            structured["training"] = training_metrics
        return structured

    def _flush(self, *, final: bool = False) -> None:
        if not self.current_stats:
            return
        self.writer.write_chunk(self.current_stats, final=final)
        if self.verbose:
            label = "final" if final else f"batch ({len(self.current_stats)} entries)"
            print(f"[StatsCallback] wrote {label} to {self.save_path}")
        self.current_stats = []


def _normalise_stat(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _normalise_stat(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalise_stat(item) for item in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload
