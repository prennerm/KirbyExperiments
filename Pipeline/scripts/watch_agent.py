#!/usr/bin/env python3
"""Lässt einen trainierten Kirby-Agenten live spielen."""
from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from kirby_pipeline.train import (
    build_vector_env,
    load_config,
    prepare_model_config,
    select_model_class,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kirby-Agent beobachten.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Pfad zur YAML-Konfiguration oder zu config_resolved.yaml (optional).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Pfad zu einer .zip-Datei, einem checkpoints/-Ordner oder dem Variant-Run-Root. "
            "Wenn ein Verzeichnis übergeben wird, wird automatisch der jüngste Run bzw. das jüngste Checkpoint gewählt."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Policy deterministisch auswerten (Standard: stochastisch).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless laufen lassen (kein Fenster).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Anzahl Schritte bevor automatisch gestoppt wird (Default 5000).",
    )
    parser.add_argument(
        "--emulation-speed",
        type=int,
        default=1,
        help="PyBoy emulation_speed (1 = Echtzeit).",
    )
    parser.add_argument(
        "--print-rewards",
        action="store_true",
        help="Rewards ausgeben, sobald sie auftreten.",
    )
    return parser.parse_args()


def build_env(
    cfg: Dict[str, Any],
    run_dir: Path,
    *,
    headless: bool,
    emulation_speed: int,
    print_rewards: bool,
) -> Any:
    env_conf = dict(cfg.get("env", {}))
    env_conf["headless"] = headless
    env_conf["emulation_speed"] = emulation_speed
    env_conf["print_rewards"] = print_rewards
    module_name = env_conf.pop("module", cfg.get("env", {}).get("module", "kirby_gym_env"))
    class_name = env_conf.pop("class", cfg.get("env", {}).get("class", "KirbyGymEnv"))
    seed = int(cfg.get("seed", 0))
    return build_vector_env(env_conf, module_name, class_name, num_cpu=1, seed=seed, run_dir=run_dir)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def _handle_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[watch_agent] Stop angefordert – schließe nach aktuellem Schritt ...")


signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)


def resolve_checkpoint(path: Path) -> Tuple[Path, Path]:
    path = path.resolve()
    if path.is_file():
        return path, path.parent.parent
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint-Pfad nicht gefunden: {path}")

    if path.name == "checkpoints":
        candidate_dir = path
        run_dir = path.parent
    elif (path / "checkpoints").is_dir():
        candidate_dir = path / "checkpoints"
        run_dir = path
    else:
        run_dirs = [p for p in path.iterdir() if p.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"Keine Run-Ordner in {path}")
        run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
        candidate_dir = run_dir / "checkpoints"

    zips = [p for p in candidate_dir.glob("*.zip")]
    if not zips:
        raise FileNotFoundError(f"Keine Checkpoints in {candidate_dir}")
    best = max(zips, key=lambda p: p.stat().st_mtime)
    return best, run_dir


def resolve_latest_run(session_root: Path) -> Path:
    if not session_root.exists():
        raise FileNotFoundError(f"Session root existiert nicht: {session_root}")
    run_dirs = [p for p in session_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"Keine Runs in {session_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def infer_checkpoint_from_config(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    session_root = cfg.get("paths", {}).get("session_root", "experiments/kirby")
    session_root_path = Path(session_root)
    if not session_root_path.is_absolute():
        session_root_path = (PROJECT_ROOT / session_root_path).resolve()
    run_dir = resolve_latest_run(session_root_path)
    checkpoints_dir = run_dir / "checkpoints"
    zips = [p for p in checkpoints_dir.glob("*.zip")]
    if not zips:
        raise FileNotFoundError(f"Keine Checkpoints in {checkpoints_dir}")
    best = max(zips, key=lambda p: p.stat().st_mtime)
    return best, run_dir


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    resolved = run_dir / "config_resolved.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} nicht gefunden – Bitte --config angeben.")
    return load_config(resolved)


def main() -> None:
    args = parse_args()
    if not args.config and not args.checkpoint:
        raise SystemExit("Bitte mindestens --config oder --checkpoint angeben.")

    checkpoint_path: Path | None = None
    run_dir: Path | None = None
    cfg: Dict[str, Any] | None = None

    if args.checkpoint:
        checkpoint_path, run_dir = resolve_checkpoint(args.checkpoint)
        print(f"Resolved checkpoint (via --checkpoint): {checkpoint_path}")

    if args.config:
        cfg = load_config(args.config)
        if checkpoint_path is None:
            checkpoint_path, run_dir = infer_checkpoint_from_config(cfg)
            print(f"Resolved checkpoint (via config): {checkpoint_path}")
    else:
        if run_dir is None:
            raise SystemExit("Konnte Run nicht bestimmen – bitte --config oder gültiges --checkpoint angeben.")
        cfg = load_run_config(run_dir)

    assert checkpoint_path is not None and run_dir is not None and cfg is not None

    vec_env = build_env(
        cfg,
        run_dir,
        headless=args.headless,
        emulation_speed=args.emulation_speed,
        print_rewards=args.print_rewards,
    )

    model_cfg = prepare_model_config(cfg.get("model", {}), cfg.get("env", {}), num_cpu=1)
    model_type = model_cfg["type"]
    ModelCls = select_model_class(model_type)
    model = ModelCls.load(str(checkpoint_path), env=vec_env)

    obs = vec_env.reset()
    state = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    total_steps = 0

    print(
        f"Watching {checkpoint_path.name} | deterministic={args.deterministic} | "
        f"headless={args.headless} | max_steps={args.max_steps}"
    )

    print_rewards = args.print_rewards

    try:
        while not STOP_REQUESTED and (args.max_steps <= 0 or total_steps < args.max_steps):
            action, state = model.predict(
                obs,
                state=state,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            obs, rewards, dones, infos = vec_env.step(action)
            total_steps += 1

            if print_rewards:
                for reward in rewards:
                    if reward != 0:
                        print(f"[step {total_steps}] reward={reward:.3f}")

            episode_starts = dones
            if dones.any():
                state = None

    except KeyboardInterrupt:
        print("\n[watch_agent] Abgebrochen durch Benutzer.")
    finally:
        vec_env.close()
        print(f"[watch_agent] Fertig nach {total_steps} Schritten.")


if __name__ == "__main__":
    main()
