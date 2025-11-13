#!/usr/bin/env python3
"""
create_init_state_simple.py
Simple interactive script to create init state for Kirby training.

Navigate to Level 1 start, press Ctrl+C to save.
"""
from pyboy import PyBoy
from pathlib import Path

rom_path = Path("data/kirbys_dream_land.gb")
state_path = Path("data/kirby_init.state")

print("=" * 60)
print("Kirby Init State Creator")
print("=" * 60)
print("\nControls:")
print("  Arrow Keys:  Move Kirby")
print("  Z:          A Button (Jump/Confirm)")
print("  X:          B Button (Inhale)")
print("  Enter:      Start")
print("  Backspace:  Select")
print("\nInstructions:")
print("  1. Wait for title screen")
print("  2. Press Enter at 'START/OPTION'")
print("  3. Press Enter on 'NORMAL' difficulty")
print("  4. Wait for Kirby to spawn in Level 1")
print("  5. Press Ctrl+C to save state")
print("=" * 60)
print(f"\nState will be saved to: {state_path}")

input("\nPress Enter to start PyBoy...")

pyboy = PyBoy(str(rom_path), window="SDL2")
pyboy.set_emulation_speed(1)  # Real-time

print("\n✅ PyBoy started")
print("   Navigate to Level 1 start, then press Ctrl+C")

try:
    while True:
        pyboy.tick()
except KeyboardInterrupt:
    print("\n\n💾 Saving state...")
    with open(state_path, "wb") as f:
        pyboy.save_state(f)
    pyboy.stop()

    print("=" * 60)
    print("✅ Init state saved successfully!")
    print("=" * 60)
    print(f"Location: {state_path}")
    print("\nTo use in training:")
    print("  1. Edit configs/kirby/base.yaml")
    print("  2. Change: init_state: 'data/kirby_init.state'")
    print("  3. Agent will start directly in gameplay")
    print("=" * 60)
