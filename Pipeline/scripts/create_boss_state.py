#!/usr/bin/env python3
"""
create_boss_state.py
Create savestate at the start of boss fight (Whispy Woods).
"""
from pyboy import PyBoy
from pathlib import Path

rom_path = Path("data/kirbys_dream_land.gb")
level1_state = Path("data/kirby_init.state")
boss_state = Path("data/kirby_boss.state")

print("=" * 60)
print("Boss Fight Savestate Creator")
print("=" * 60)
print("\nThis creates a savestate at Level 1 boss (Whispy Woods)")
print(f"Will be saved to: {boss_state}")
print("\nControls:")
print("  Arrow Keys: Move")
print("  Y (or Z): Jump")
print("  X: Inhale")
print("  ESC: Close window when ready")
print("=" * 60)

input("\nPress Enter to start...")

pyboy = PyBoy(str(rom_path), window="SDL2")
pyboy.set_emulation_speed(1)

# Load Level 1 start if available
if level1_state.exists():
    with open(level1_state, "rb") as f:
        pyboy.load_state(f)
    print("\n✅ Loaded Level 1 state")
    print("   Play through Level 1 to reach Whispy Woods boss")
else:
    print("\n⚠️  No Level 1 state - starting from ROM")
    print("   Navigate menus, then play to boss")

print("\nInstructions:")
print("1. Play through Level 1")
print("2. Reach the boss door (tree area at end)")
print("3. Position Kirby RIGHT BEFORE boss fight starts")
print("4. Close PyBoy window (ESC or X button)")
print("\n💡 TIP: Boss spawns when you walk far enough right")
print("        Stop JUST before he appears!")

try:
    while pyboy.tick():
        pass
except KeyboardInterrupt:
    pass

print("\n💾 Saving boss state...")
with open(boss_state, "wb") as f:
    pyboy.save_state(f)

pyboy.stop()

print("=" * 60)
print("✅ Boss state saved!")
print("=" * 60)
print(f"Location: {boss_state}")
print("\nYou now have TWO states:")
print(f"  - {level1_state.name}: Level 1 start (platforming)")
print(f"  - {boss_state.name}: Boss fight start")
print("\nUse boss state for focused boss training later!")
print("=" * 60)
