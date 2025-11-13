# Kirby's Dream Land - Complete Setup Guide

## Table of Contents
1. [ROM Setup](#rom-setup)
2. [Environment Setup](#environment-setup)
3. [First Steps](#first-steps)
4. [Reference Code](#reference-code)
5. [RAM Addresses](#ram-addresses)
6. [Reward Design](#reward-design)

---

## ROM Setup

### Step 1: Obtain ROM

**Legal Notice:** You should own a physical copy of Kirby's Dream Land to legally use the ROM.

**ROM Details:**
- **Game:** Kirby's Dream Land (U.S.A. & Europe)
- **Platform:** Game Boy (original, not Color)
- **File Size:** 256 KiB (262,144 bytes)
- **ROM Checksum:** `0xADF9` (from GB header at 0x014E-0x014F)
- **Header Checksum:** `0x98` (from GB header at 0x014D)
- **Source:** [Data Crystal - Kirby's Dream Land](https://datacrystal.tcrf.net/wiki/Kirby%27s_Dream_Land)

**Download Sources:**
- Archive.org: Game Boy ROM collections
- Emulation communities (check legality in your region)

### Step 2: Place ROM

```bash
cp /path/to/kirbys_dream_land.gb data/
```

**Expected path:** `data/kirbys_dream_land.gb`

### Step 3: Verify Checksum

```bash
python scripts/verify_rom.py
```

**Expected output:**
```
✅ ROM found: data/kirbys_dream_land.gb
✅ File size matches: 262,144 bytes
MD5 Checksum: [your ROM's MD5]
Game Boy Header Verification:
   Header Checksum: 0x98 (expected: 0x98)
   ROM Checksum:    0xADF9 (expected: 0xADF9)
✅ All checksums match! ROM verified.
```

---

## Environment Setup

### Prerequisites

- Python 3.10+
- Conda (Anaconda/Miniconda)
- Git
- CUDA 11.8+ (for GPU training, optional but recommended)

### Installation

```bash
# 1. Clone/navigate to project
cd KirbyExperiments/Pipeline

# 2. Create conda environment
conda env create -f environment.yml
conda activate kirby_env

# 3. Install package in editable mode
pip install -e .

# 4. Verify installation
python -c "import kirby_pipeline; print('✅ Package installed')"
python -c "from pyboy import PyBoy; print('✅ PyBoy installed')"
python -c "from stable_baselines3 import PPO; print('✅ SB3 installed')"
```

---

## First Steps

### Test 1: Manual Play (5 minutes)

Test if you can play Kirby manually with keyboard:

```bash
python scripts/test_manual_play.py
```

**Controls:**
- Arrow Keys: Move Kirby
- Z: A Button (Jump/Confirm)
- X: B Button (Inhale/Attack)
- Enter: Start
- Backspace: Select
- Ctrl+C: Quit

**What to check:**
- ✅ ROM loads without errors
- ✅ You see Kirby's Dream Land title screen
- ✅ Kirby responds to arrow keys
- ✅ Jump (Z) works
- ✅ Inhale (X) works

---

## Reference Code

When implementing Kirby environments, refer to these Pokemon Pipeline files:

### Core Environment Structure

**File:** `../../PokemonRedExperiments/Pipeline/src/poke_pipeline/red_gym_env_v2.py`

**What to learn:**
- Gym API implementation (`reset()`, `step()`, `render()`)
- Observation space setup (screen as CNN input)
- Action space definition (discrete buttons)
- Reward shaping logic
- Episode termination conditions

**Key sections:**
```python
# Lines 50-100: Observation/Action spaces
# Lines 150-250: reset() implementation
# Lines 300-450: step() logic
# Lines 500-600: Reward calculation
```

### LSTM-Specific Environment

**File:** `../../PokemonRedExperiments/Pipeline/src/poke_pipeline/red_gym_env_lstm.py`

**What to learn:**
- Differences for RecurrentPPO
- State management for LSTM
- Frame handling without stacking

### Logging & Callbacks

**File:** `../../PokemonRedExperiments/Pipeline/src/poke_pipeline/callbacks.py`

**What to adapt:**
- JSON logging structure
- Position tracking (change from map_id/x/y to level/x_pos)
- Reward component logging
- Stats aggregation

**Lines to modify:**
```python
# Line 121-161: JSON log structure
# Change: "map_id", "x", "y" → "level", "x_pos", "y_pos"
# Change: "badges", "seen_pokemon" → "score", "lives"
```

### Lambda Discrepancy (Copy 1:1)

**File:** `../../PokemonRedExperiments/Pipeline/src/poke_pipeline/ppo_lambda_discrepancy.py`

**Action:** Copy this file unchanged to `src/kirby_pipeline/`

```bash
cp ../../PokemonRedExperiments/Pipeline/src/poke_pipeline/ppo_lambda_discrepancy.py \
   src/kirby_pipeline/
```

### Training Script Template

**File:** `../../PokemonRedExperiments/Pipeline/src/poke_pipeline/train.py`

**What to adapt:**
- Minimal changes needed (dynamic environment loading already supports Kirby)
- Config file paths: `configs/kirby/` instead of `configs/`
- Experiment paths: `experiments/kirby/` instead of `experiments/v1/`

### Sanity Check Examples

**Directory:** `../../PokemonRedExperiments/Pipeline/src/poke_pipeline/sanity_check/`

**Files to reference:**
- `cartpole_test.py`: Basic Gym Env structure
- `lunar_lander_test.py`: Observation/Action space examples

---

## RAM Addresses

### Discovery Process

RAM addresses for Kirby's Dream Land need to be found through:

1. **Community Resources**
   - Check speedrun communities (speedrun.com forums)
   - Search GitHub for "Kirby Dream Land RAM map"
   - Check gbdev.io forums

2. **Manual Discovery** (if needed)
   - Use PyBoy memory viewer
   - Play game manually, watch memory change
   - Document findings in `RAM_ADDRESSES.md`

### Required Addresses

We need to find memory locations for:

- **Score** (4 bytes, BCD encoded)
- **Lives** (1 byte)
- **X Position** (2 bytes, world coordinate)
- **Y Position** (2 bytes, optional)
- **Current Level** (1 byte, 0-4 for 5 levels)
- **Health/Power** (1 byte, optional)
- **Boss Defeated Flags** (bitmask, optional)

### Typical Game Boy RAM Regions

```
0xC000-0xDFFF: Work RAM (8KB)
  → Most game state stored here
  → Score, lives, position usually in 0xC000-0xC100

0xFF00-0xFF7F: I/O Registers
  → Don't use for game state

0xFE00-0xFE9F: OAM (Sprite data)
  → Could show player sprite position
```

**Action:** Once found, document in `RAM_ADDRESSES.md`

---

## Reward Design

### Kirby-Specific Reward Components

Unlike Pokemon (sparse rewards, badges), Kirby has dense rewards:

```python
reward = (
    score_delta * 0.01         # Points gained (frequent, small)
    + x_progress * 0.1         # Moving right (dense)
    + level_complete * 1000    # Finishing level (sparse, huge)
    - death_penalty * 100      # Dying (sparse, negative)
    - time_penalty * 0.01      # Efficiency (constant small negative)
)
```

### Rationale

1. **Score Delta (0.01x)**
   - Encourages collecting items, defeating enemies
   - Dense signal (every few seconds)
   - Small weight to avoid dominating

2. **X Progress (0.1x)**
   - Primary goal: move right
   - Dense signal (every step)
   - Higher weight than score

3. **Level Complete (1000)**
   - Milestone achievement
   - Sparse but critical
   - Large reward to guide long-term planning

4. **Death Penalty (-100)**
   - Discourage risky behavior
   - Moderate penalty (not too harsh)
   - Balances with level completion reward

5. **Time Penalty (-0.01)**
   - Prevents "standing around"
   - Very small, just breaks ties
   - Encourages efficiency

### Comparison to Pokemon

| Aspect | Pokemon | Kirby |
|--------|---------|-------|
| **Primary Goal** | Badges (sparse) | Score + Level Progress (dense) |
| **Episode Length** | Hours | Minutes |
| **Reward Frequency** | Rare events | Frequent small rewards |
| **Exploration** | Open world | Linear levels |

**Conclusion:** Kirby is much easier to learn due to dense rewards!

---

## Next Steps

After completing setup:

1. ✅ **Verify ROM** works
2. ✅ **Manual play test** successful
3. ⏳ **Find RAM addresses** (see `RAM_ADDRESSES.md`)
4. ⏳ **Implement `kirby_gym_env.py`** (see reference code)
5. ⏳ **Run sanity checks** (random agent test)
6. ⏳ **Start training k_v1** (PPO baseline)

Track progress in `TODO.md`.

---

## Troubleshooting

### ROM Won't Load

**Error:** `FileNotFoundError: No such file`
- **Fix:** Check path is exactly `data/kirbys_dream_land.gb`

**Error:** `Unsupported ROM type`
- **Fix:** Ensure it's GB (not GBC/GBA), download correct version

### PyBoy Window Not Showing

**Error:** `SDL2 not found`
- **Fix:** `conda install -c conda-forge sdl2`

### Manual Controls Don't Work

**Issue:** Buttons not responding
- **Fix:** ROM might be wrong region (try USA or Japan version)
- **Fix:** Ensure window is focused (click on it)

---

## Support

- **Issues:** Create GitHub issue
- **Questions:** Check `docs/LEARNINGS.md` for common pitfalls
- **Pokemon Reference:** See `../../PokemonRedExperiments/Pipeline/CLAUDE.md`
