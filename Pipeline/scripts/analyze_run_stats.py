#!/usr/bin/env python3
"""
Aggregiert Statistiken aus Kirby-SB3-Logs.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_stats(run_dir: Path) -> pd.DataFrame:
    pattern = run_dir / "logs" / "stats_*.csv"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien unter {pattern}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    sort_key = "step" if "step" in df.columns else "total_steps"
    df = df.sort_values(sort_key).reset_index(drop=True)
    return df


def _fmt(series: pd.Series | None, func, fmt: str) -> str:
    if series is None:
        return "n/a"
    value = func(series)
    return fmt.format(value)


def summarize_run(run_dir: Path, df: pd.DataFrame, num_bins: int) -> str:
    summary: List[str] = []
    summary.append(f"Run: {run_dir}")
    summary.append(f"Zeilen: {len(df):,}")
    summary.append(f"Global max progress: {df['progress.level'].max():.0f}")
    summary.append(f"Global min reward_step: {df['reward.step'].min():.4f}")
    summary.append(f"Global max reward_total: {df['reward.total'].max():.2f}")
    summary.append(f"Warpstar Hits: {int(df['status.warpstar'].sum())}")

    optional_globals = [
        ("clip_fraction", "Clip fraction Ø={mean:.3f} / max={max:.3f}"),
        ("entropy_loss", "Entropy loss Ø={mean:.2f} / min={min:.2f} / max={max:.2f}"),
        ("training.loss.total", "Training loss Ø={mean:.3f}"),
        ("training.policy_loss", "Policy loss Ø={mean:.5f}"),
        ("training.mc_loss", "Value loss Ø={mean:.3f}"),
        ("training.ld_loss", "LD loss Ø={mean:.5f}"),
        ("training.approx_kl", "Approx KL Ø={mean:.5f}"),
    ]
    for column, template in optional_globals:
        if column in df.columns:
            series = df[column]
            summary.append(
                template.format(
                    mean=series.mean(),
                    max=series.max(),
                    min=series.min(),
                )
            )

    sort_key = "step" if "step" in df.columns else "total_steps"
    total_last = df[sort_key].iloc[-1]
    bins = np.array_split(df, num_bins)

    for idx, bin_df in enumerate(bins, 1):
        if bin_df.empty:
            continue
        pct = (bin_df[sort_key].iloc[-1] / total_last) * 100 if total_last else 0
        clip_series = bin_df["clip_fraction"] if "clip_fraction" in bin_df.columns else None
        entropy_series = bin_df["entropy_loss"] if "entropy_loss" in bin_df.columns else None
        train_loss_series = bin_df["training.loss.total"] if "training.loss.total" in bin_df.columns else None
        train_policy_series = (
            bin_df["training.policy_loss"] if "training.policy_loss" in bin_df.columns else None
        )
        line = (
            f"Bin {idx:02d} (~{pct:.1f}%): "
            f"progress_mean={bin_df['progress.level'].mean():.1f}, "
            f"progress_min={bin_df['progress.level'].min():.0f}, "
            f"progress_max={bin_df['progress.level'].max():.0f}, "
            f"reward_step_med={bin_df['reward.step'].median():.4f}, "
            f"reward_step_min={bin_df['reward.step'].min():.4f}, "
            f"reward_step_max={bin_df['reward.step'].max():.4f}, "
            f"reward_total_mean={bin_df['reward.total'].mean():.2f}, "
            f"clip_frac_mean={_fmt(clip_series, pd.Series.mean, '{:.3f}')}, "
            f"clip_frac_max={_fmt(clip_series, pd.Series.max, '{:.3f}')}, "
            f"entropy_mean={_fmt(entropy_series, pd.Series.mean, '{:.2f}')}, "
            f"entropy_min={_fmt(entropy_series, pd.Series.min, '{:.2f}')}, "
            f"entropy_max={_fmt(entropy_series, pd.Series.max, '{:.2f}')}, "
            f"train_loss_mean={_fmt(train_loss_series, pd.Series.mean, '{:.3f}')}, "
            f"policy_loss_mean={_fmt(train_policy_series, pd.Series.mean, '{:.5f}')}, "
            f"warpstar_hits={int(bin_df['status.warpstar'].sum())}"
        )
        summary.append(line)
    summary.append("")
    return "\n".join(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kirby-Logs analysieren.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run-Verzeichnisse (mit logs/).")
    parser.add_argument("--bins", type=int, default=10, help="Anzahl Intervalle pro Run.")
    args = parser.parse_args()

    for run_path in args.runs:
        run_dir = Path(run_path).resolve()
        try:
            df = load_stats(run_dir)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        summary_text = summarize_run(run_dir, df, args.bins)
        print(summary_text)
        out_file = run_dir / "statistics.txt"
        out_file.write_text(summary_text, encoding="utf-8")
        print(f"[INFO] Statistik gespeichert in {out_file}\n")


if __name__ == "__main__":
    main()
