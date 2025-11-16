#!/usr/bin/env python3
"""Training entry point for Kirby''s Dream Land."""
from __future__ import annotations

import argparse
import importlib
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO

from kirby_pipeline.callbacks import StatsCallback
from kirby_pipeline.ppo_lambda_discrepancy import CnnLstmPolicyLD, RecurrentPPOLD

MODEL_REGISTRY = {
    "PPO": PPO,
    "RecurrentPPO": RecurrentPPO,
    "RecurrentPPOLD": RecurrentPPOLD,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kirby agents with Stable-Baselines3")
    parser.add_argument("--variant", required=True, help="Configuration variant name (e.g. k_v1)")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=Path, help="Optional checkpoint to resume from")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = _deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _load(current_path: Path) -> Dict[str, Any]:
        data = yaml.safe_load(current_path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {current_path} must define a mapping")
        extends = data.pop("extends", None)
        if extends:
            merged: Dict[str, Any] = {}
            extends_list = [extends] if isinstance(extends, str) else list(extends)
            for entry in extends_list:
                base_path = (current_path.parent / entry).resolve()
                merged = _deep_update(merged, _load(base_path))
            return _deep_update(merged, data)
        return data

    return _load(path.resolve())


def make_run_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "root": base,
        "logs": base / "logs",
        "checkpoints": base / "checkpoints",
        "tensorboard": base / "tensorboard",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def make_env_factory(
    module_name: str,
    class_name: str,
    env_conf: Dict[str, Any],
    rank: int,
    seed: int,
    env_root: Path,
) -> Callable[[], Any]:
    module = importlib.import_module(f"kirby_pipeline.{module_name}")
    EnvCls = getattr(module, class_name)

    def _init():
        worker_conf = dict(env_conf)
        worker_conf["worker_rank"] = rank
        worker_path = env_root / f"worker_{rank:02d}"
        worker_path.mkdir(parents=True, exist_ok=True)
        worker_conf["session_path"] = str(worker_path)
        env = EnvCls(worker_conf)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vector_env(
    env_conf: Dict[str, Any],
    module_name: str,
    class_name: str,
    num_cpu: int,
    seed: int,
    run_dir: Path,
):
    env_root = run_dir / "env"
    env_root.mkdir(parents=True, exist_ok=True)
    factories = [
        make_env_factory(module_name, class_name, env_conf, idx, seed, env_root)
        for idx in range(num_cpu)
    ]
    if num_cpu > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv([factories[0]])


def select_model_class(model_type: str):
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model type: {model_type}") from exc


def prepare_model_config(model_cfg: Dict[str, Any], env_conf: Dict[str, Any], num_cpu: int) -> Dict[str, Any]:
    cfg = dict(model_cfg)
    type_name = cfg.pop("type", "PPO")
    policy = cfg.pop("policy", None)
    max_steps = int(env_conf.get("max_steps", 1024))
    if "n_steps" not in cfg:
        calculated = max(1, max_steps // max(1, num_cpu))
        cfg["n_steps"] = min(calculated, 2048)
    if policy is None:
        if type_name == "RecurrentPPOLD":
            policy = CnnLstmPolicyLD
        elif type_name == "RecurrentPPO":
            policy = "CnnLstmPolicy"
        else:
            policy = "CnnPolicy"
    elif isinstance(policy, str) and policy == "CnnLstmPolicyLD":
        policy = CnnLstmPolicyLD
    return {"policy": policy, "type": type_name, "kwargs": cfg}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    num_cpu = max(1, int(cfg.get("num_cpu", 1)))
    total_timesteps = int(cfg.get("total_timesteps", 1_000_000))

    session_root = Path(cfg.get("paths", {}).get("session_root", "experiments/kirby"))
    session_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = cfg.get("run_label")
    if run_label:
        run_id = f"{run_id}_{run_label}"
    run_dir = session_root / run_id
    dirs = make_run_dirs(run_dir)

    env_conf = dict(cfg.get("env", {}))
    env_conf.setdefault("session_path", str(run_dir / "env"))
    env_conf["num_cpu"] = num_cpu

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_module = env_conf.pop("module", cfg.get("env", {}).get("module", "kirby_gym_env"))
    env_class = env_conf.pop("class", cfg.get("env", {}).get("class", "KirbyGymEnv"))

    vec_env = build_vector_env(env_conf, env_module, env_class, num_cpu, seed, run_dir)

    model_cfg = prepare_model_config(cfg.get("model", {}), env_conf, num_cpu)
    model_type = model_cfg["type"]
    policy = model_cfg["policy"]
    model_kwargs = model_cfg["kwargs"]
    ModelCls = select_model_class(model_type)

    tensorboard_log = str(dirs["tensorboard"])
    if args.resume:
        model = ModelCls.load(str(args.resume), env=vec_env)
        model.tensorboard_log = tensorboard_log
    else:
        model = ModelCls(policy, vec_env, tensorboard_log=tensorboard_log, **model_kwargs)

    stats_callback = StatsCallback(
        save_freq=int(cfg.get("save_freq_stats", 100)),
        save_path=str(dirs["logs"]),
        output_format=cfg.get("logging", {}).get("format", "csv"),
        structured=cfg.get("logging", {}).get("structured", True),
        verbose=int(cfg.get("logging", {}).get("verbose", 0)),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg.get("save_freq", max(1, model_kwargs.get("n_steps", 1024)))),
        save_path=str(dirs["checkpoints"]),
        name_prefix=args.variant,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = CallbackList([checkpoint_callback, stats_callback])

    config_snapshot = run_dir / "config_resolved.yaml"
    config_snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"[train] Starting run {run_dir} with {num_cpu} env(s) for {total_timesteps} steps")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        print("[train] Interrupted by user, saving partial results...")
    finally:
        final_path = dirs["checkpoints"] / f"{args.variant}_final"
        model.save(str(final_path))
        vec_env.close()
        print(f"[train] Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
