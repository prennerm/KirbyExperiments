#!/usr/bin/env python3
"""
manually launch Kirby in PyBoy for direct keyboard control.
"""
from pathlib import Path
from pyboy import PyBoy

ROM_PATH = Path("data/kirbys_dream_land.gb")
STATE_PATH = Path("data/kirby_init.state")

print("Kirby manual play demo")
print("Keyboard:")
print("  Arrow keys  -> Move")
print("  Z (or S on DE layout) -> A (Jump/Inhale)")
print("  X           -> B")
print("  Enter       -> Start")
print("  Backspace   -> Select")
print("  Esc         -> Close window")

if not ROM_PATH.exists():
    raise FileNotFoundError(f"ROM not found: {ROM_PATH}")

pyboy = PyBoy(str(ROM_PATH), window="SDL2")
pyboy.set_emulation_speed(1)

if STATE_PATH.exists():
    with STATE_PATH.open("rb") as fh:
        pyboy.load_state(fh)
    print("Loaded init state")
else:
    print("No init state found; starting from title screen")

print("PyBoy running. Play the game, close the window to exit.")
try:
    while pyboy.tick():
        pass
except KeyboardInterrupt:
    pass
finally:
    pyboy.stop()