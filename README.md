# Kirby's Dream Land RL Training Pipeline

Reinforcement Learning experiments for Kirby's Dream Land (Game Boy) to validate Lambda Discrepancy method.

## Project Overview

This project implements and evaluates different RL approaches on Kirby's Dream Land:
- **k_v1:** Baseline PPO
- **k_v3:** RecurrentPPO with LSTM
- **k_v4:** RecurrentPPO + Lambda Discrepancy

**Purpose:** Kirby provides a faster, simpler testbed compared to Pokemon Red for validating the Lambda Discrepancy algorithm.

## Repository Structure

```
KirbyExperiments/
├── Pipeline/               # Main training pipeline
│   ├── src/
│   │   └── kirby_pipeline/
│   │       └── kirby_gym_env.py
│   ├── configs/
│   │   └── kirby/
│   │       ├── base.yaml
│   │       ├── k_v1.yaml
│   │       ├── k_v3.yaml
│   │       └── k_v4.yaml
│   ├── scripts/
│   ├── docs/
│   └── experiments/       # Training runs (not in git)
└── README.md
```

## Quick Start

See [Pipeline/README.md](Pipeline/README.md) for detailed setup instructions.

```bash
# 1. Setup environment
cd Pipeline
conda env create -f environment.yml
conda activate kirby_env
pip install -e .

# 2. Verify ROM (you must provide your own ROM)
python scripts/verify_rom.py

# 3. Test environment
python scripts/test_kirby_env.py

# 4. Start training
python -m kirby_pipeline.train --config configs/kirby/k_v1.yaml
```

## Documentation

- **[KIRBY_SETUP.md](Pipeline/docs/KIRBY_SETUP.md)** - Complete setup guide
- **[RAM_ADDRESSES.md](Pipeline/docs/RAM_ADDRESSES.md)** - Game Boy memory map
- **[INIT_STATE_GUIDE.md](Pipeline/docs/INIT_STATE_GUIDE.md)** - Creating training savestates

## Related Projects

This is a companion project to [PokemonRedExperiments](https://github.com/YOUR_USERNAME/PokemonRedExperiments):
- **Pokemon Red:** Complex, long-term training (2B+ steps)
- **Kirby:** Simple, fast validation (~50M steps)

## Legal Notice

You must own a physical copy of Kirby's Dream Land to legally use the ROM file. ROM files are **not included** in this repository.

## License

[Add your license here]

## Citation

If you use this code for research, please cite:

```
[Your thesis citation]
```
