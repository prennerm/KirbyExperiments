#!/usr/bin/env python3
"""
test_init_state.py
Verify that the init state loads correctly.
"""
from pyboy import PyBoy
from pathlib import Path

state_path = Path("data/kirby_init.state")
rom_path = Path("data/kirbys_dream_land.gb")

print("=" * 60)
print("Init State Verification")
print("=" * 60)

if not state_path.exists():
    print("❌ No init state found!")
    print(f"   Expected: {state_path}")
    exit(1)

print(f"✅ Init state found: {state_path}")
print(f"   Size: {state_path.stat().st_size / 1024:.1f} KB")

print("\nLoading ROM and state...")
pyboy = PyBoy(str(rom_path), window="SDL2")
pyboy.set_emulation_speed(1)

with open(state_path, "rb") as f:
    pyboy.load_state(f)

print("✅ Init state loaded successfully!")
print("\n" + "=" * 60)
print("Verification Check:")
print("=" * 60)
print("You should see:")
print("  - Kirby at the start of Green Greens (Level 1)")
print("  - Score: 0 (or very low)")
print("  - Lives: Full (usually 3)")
print("  - No title screen or menus")
print("\nIf this looks correct, init state is ready for training!")
print("Press Ctrl+C to exit")
print("=" * 60)

try:
    while True:
        pyboy.tick()
except KeyboardInterrupt:
    print("\n\n✅ Init state verification complete!")
    pyboy.stop()
