# Creating Init State for Kirby Training

## Why Do We Need This?

Without an init state, the agent starts from ROM boot every episode:
- Nintendo logo (2 seconds)
- HAL Laboratory logo (2 seconds)
- Title screen (waiting for player input)
- Difficulty select menu

**Problem:** Agent wastes training steps learning to navigate menus instead of gameplay.

**Solution:** Save a "savestate" right when gameplay starts (Kirby spawns in Level 1).

---

## Method 1: Python Script (Simplest)

### Step 1: Run Interactive Script

```bash
cd C:\Users\marwi\Documents\fhstp\masterarbeit\KirbyExperiments\Pipeline
conda activate kirby_env
python scripts/create_init_state_interactive.py
```

### Step 2: Follow Instructions

The script will:
1. Open PyBoy window
2. Let you play through menus
3. Automatically save state after 60 seconds

### Step 3: Manual Controls

While script is running:
- **Arrow Keys:** Move Kirby
- **Z:** A Button (Jump/Confirm)
- **Enter:** Start

Navigate:
1. Wait for title screen (~4 seconds)
2. Press **Enter** at "START/OPTION" screen
3. Press **Enter** on "NORMAL" difficulty
4. Kirby spawns in Green Greens
5. **Wait for script to auto-save** (60 seconds total)

---

## Method 2: Manual PyBoy (More Control)

### Step 1: Install PyBoy GUI (if not already)

PyBoy is already installed in your `kirby_env`.

### Step 2: Create Simple Loader Script

Create `scripts/load_rom_and_wait.py`:

```python
from pyboy import PyBoy
from pathlib import Path
import sys

rom_path = Path("data/kirbys_dream_land.gb")
output_path = Path("data/kirby_init.state")

print("Loading ROM...")
print("Navigate to Level 1 start, then press Ctrl+C")
print(f"State will be saved to: {output_path}")

pyboy = PyBoy(str(rom_path), window="SDL2")
pyboy.set_emulation_speed(1)

try:
    while True:
        pyboy.tick()
except KeyboardInterrupt:
    print("\nSaving state...")
    with open(output_path, "wb") as f:
        pyboy.save_state(f)
    print(f"✅ Saved to {output_path}")
    pyboy.stop()
```

### Step 3: Run and Save

```bash
python scripts/load_rom_and_wait.py
```

1. Play through menus to Level 1 start
2. Press **Ctrl+C** when ready
3. State automatically saves

---

## Method 3: Use test_manual_play.py (Existing Script)

### Modify Existing Script

You already have `scripts/test_manual_play.py`. Let's add save functionality:

```bash
# Edit test_manual_play.py to add save on Ctrl+C
```

Actually, let me create a better version...

---

## Method 4: Quick and Dirty (Recommended for Now)

### Just Use Pokemon's Approach

Since we're testing, you can **skip init state** for now:

**In `configs/kirby/base.yaml`:**
```yaml
env:
  init_state: ""  # Empty = start from ROM boot
```

**Why this is OK for testing:**
- Agent learns to press Start in menu (takes ~100 episodes)
- Once learned, always does it correctly
- Not a big waste of training time
- **We can add init state later** when we're sure environment works

---

## Creating Init State - Final Recommendation

### Easiest Method: Extend test_manual_play.py

I'll create a version that saves state on exit:

```python
# scripts/create_init_state_simple.py
from pyboy import PyBoy
from pathlib import Path

rom_path = Path("data/kirbys_dream_land.gb")
state_path = Path("data/kirby_init.state")

print("=" * 60)
print("Kirby Init State Creator")
print("=" * 60)
print("\nControls:")
print("  Arrow Keys: Move")
print("  Z: A Button (Jump/Confirm)")
print("  X: B Button (Inhale)")
print("  Enter: Start")
print("\nInstructions:")
print("  1. Navigate through menus to Level 1 start")
print("  2. Position Kirby at spawn point")
print("  3. Press Ctrl+C to save and exit")
print("=" * 60)

input("\nPress Enter to start...")

pyboy = PyBoy(str(rom_path), window="SDL2")
pyboy.set_emulation_speed(1)

try:
    while True:
        pyboy.tick()
except KeyboardInterrupt:
    print("\n\nSaving state...")
    with open(state_path, "wb") as f:
        pyboy.save_state(f)
    pyboy.stop()
    print(f"✅ State saved: {state_path}")
    print("\nTo use in training, update base.yaml:")
    print("  init_state: 'data/kirby_init.state'")
```

---

## Verification

After creating init state, test it:

```python
# scripts/test_init_state.py
from pyboy import PyBoy
from pathlib import Path

state_path = Path("data/kirby_init.state")

if not state_path.exists():
    print("❌ No init state found")
    exit(1)

pyboy = PyBoy("data/kirbys_dream_land.gb", window="SDL2")

with open(state_path, "rb") as f:
    pyboy.load_state(f)

print("✅ Init state loaded!")
print("   You should see Kirby at Level 1 start")
print("   Press Ctrl+C to exit")

try:
    while True:
        pyboy.tick()
except KeyboardInterrupt:
    pyboy.stop()
```

---

## Summary

**For Now (Testing):**
- Skip init state, let agent learn menus
- Fast enough for initial testing

**For Real Training:**
- Use Method 4 (simple script with Ctrl+C save)
- Takes 2 minutes to create
- Saves hours of training time

**Expected State:**
- Kirby at spawn point in Green Greens (Level 1)
- No enemies defeated yet
- Score: 0
- Lives: Full
- Health: Full
