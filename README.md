# Kirby's Dream Land RL Pipeline

> **Hinweis:** Die Reward-Logik (insbesondere Warpstar-Events) wird gerade überarbeitet. Aktuelle Runs dienen der Fehlersuche; Metriken können deshalb noch stark variieren.

Playground for PPO variants (feed-forward, recurrent, LD) that learn Kirby's Dream Land via PyBoy + Stable-Baselines3. The repo intentionally stays lightweight while we iterate.

## Quick Start

```bash
conda env create -f Pipeline/environment.yml
conda activate kirby_env
pip install -e Pipeline

cp /path/to/kirbys_dream_land.gb Pipeline/data/   # bring your own ROM
python Pipeline/scripts/verify_rom.py             # optional ROM sanity check

# train one variant (configs/kirby/*)
python -m kirby_pipeline.train --variant k_v1 --config configs/kirby/k_v1.yaml
```

## Repo Layout

- `Pipeline/src/kirby_pipeline/` - env, policies, training entry point
- `Pipeline/configs/kirby/` - base config plus variant overrides
- `Pipeline/experiments/kirby/` - checkpoints, CSV stats, TensorBoard logs
- `Pipeline/scripts/` - helper tools (stats analyzer, watch_agent, ROM tests)
- `Pipeline/run_kirby_batch.bat` - sequential run for k_v1 -> k_v3

## Status

Currently tuned for tens of millions of PPO steps while we refine KL handling and rewards. Expect rapid changes.
