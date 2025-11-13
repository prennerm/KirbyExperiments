# Kirby's Dream Land RL Training Pipeline

Reinforcement Learning training pipeline for Kirby's Dream Land (Game Boy) using PyBoy emulator and Stable-Baselines3.

## Project Context

This project trains RL agents to play Kirby's Dream Land using:
- **PPO** (Baseline, v1)
- **RecurrentPPO** with LSTM (v3)
- **RecurrentPPO + Lambda Discrepancy** (v4)

**Goal:** Validate Lambda Discrepancy method on a simpler, faster-training environment than Pokemon Red.

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate kirby_env

# 2. Install package
pip install -e .

# 3. Place ROM
# Download Kirby's Dream Land ROM (see docs/KIRBY_SETUP.md)
cp /path/to/kirbys_dream_land.gb data/

# 4. Verify ROM
python scripts/verify_rom.py

# 5. Test PyBoy
python scripts/test_manual_play.py

# 6. Start training
python -m kirby_pipeline.train --variant k_v1 --config configs/kirby/k_v1.yaml
```

## Project Structure

```
Pipeline/
├── src/kirby_pipeline/     # Source code
│   ├── envs/               # Gym environments
│   ├── train.py            # Training script
│   ├── run_all.py          # Agent playback
│   └── callbacks.py        # Logging
├── configs/kirby/          # Training configs
├── experiments/kirby/      # Training results
├── data/                   # ROM & save states
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## Documentation

- **[Setup Guide](docs/KIRBY_SETUP.md)** - Complete setup instructions
- **[RAM Addresses](docs/RAM_ADDRESSES.md)** - Memory locations for game state
- **[Reward Design](docs/REWARD_DESIGN.md)** - Reward shaping rationale
- **[Learnings](docs/LEARNINGS.md)** - What works/doesn't work
- **[TODO](docs/TODO.md)** - Open tasks

## Related Projects

This project is based on the [Pokemon Red RL Pipeline](../PokemonRedExperiments/Pipeline/).

Key differences:
- Simpler game (linear levels vs open world)
- Faster training (50M vs 2B+ steps)
- Different reward structure (score/progress vs badges/events)

## License

Research project for Master's Thesis. ROM not included (obtain legally).
