#!/usr/bin/env python3
"""Plot helper for Kirby stats output."""
import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of the columns {candidates} are present in stats file")


def load_stats(stats_dir: Path) -> pd.DataFrame:
    csv_files = sorted(stats_dir.glob("stats_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No stats_*.csv files found in {stats_dir}")
    frames = [pd.read_csv(path) for path in csv_files]
    df = pd.concat(frames, ignore_index=True)
    time_col = "total_steps" if "total_steps" in df.columns else "step"
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def plot_stats(df: pd.DataFrame, output_path: Path, title_suffix: Optional[str] = None) -> Path:
    time_col = "total_steps" if "total_steps" in df.columns else "step"
    level_progress = pick_column(df, ["progress.level", "level_progress"])
    boss_health = pick_column(df, ["boss.health", "boss_health"])
    warpstar_flag = pick_column(df, ["status.warpstar", "warpstar"])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    axes[0].plot(df[time_col], level_progress, label="Level progress", color="#6a5acd")
    axes[0].set_ylabel("Level progress")
    axes[0].legend(loc="upper left")

    warp_indices = df.index[warpstar_flag.astype(bool)]
    if not warp_indices.empty:
        axes[0].scatter(
            df.loc[warp_indices, time_col],
            level_progress.loc[warp_indices],
            color="orange",
            s=30,
            label="Warpstar",
        )
        axes[0].legend(loc="upper left")

    axes[1].step(df[time_col], boss_health, where="post", color="#c23b22", label="Boss health")
    axes[1].set_ylabel("Boss HP")
    axes[1].set_xlabel("Total steps")
    axes[1].legend(loc="upper right")

    title = "Kirby Stats"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Kirby stats CSV files")
    parser.add_argument("--stats-dir", required=True, help="Directory that contains stats_*.csv")
    parser.add_argument("--output", help="Optional output image path")
    parser.add_argument("--title", help="Optional title suffix")
    parser.add_argument("--show", action="store_true", help="Show plot window as well")
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists():
        raise FileNotFoundError(stats_dir)
    df = load_stats(stats_dir)

    output_path = Path(args.output) if args.output else stats_dir / "plots" / "kirby_stats.png"
    saved_path = plot_stats(df, output_path, args.title)
    print(f"[plot_kirby_stats] wrote plot to {saved_path}")

    if args.show:
        import matplotlib.image as mpimg

        img = mpimg.imread(saved_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
