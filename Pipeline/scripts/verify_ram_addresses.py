#!/usr/bin/env python3
"""
verify_ram_addresses.py
Manually verify RAM addresses from Data Crystal by observing values during gameplay.
"""
from pyboy import PyBoy
from pathlib import Path
import time

rom_path = Path("data/kirbys_dream_land.gb")
state_path = Path("data/kirby_init.state")

# RAM addresses to verify
ADDRS = {
    "Kirby X": 0xD05C,
    "Kirby Y": 0xD05D,
    "Scroll X": 0xD053,
    "Scroll Y": 0xD055,
    "Health": 0xD086,
    "Lives": 0xD089,
    "Score 100K": 0xD06F,
    "Score 10K": 0xD070,
    "Score 1K": 0xD071,
    "Score 100": 0xD072,
    "Score 10": 0xD073,
    "Game State": 0xD02C,
    "Boss HP": 0xD093,
}

print("=" * 70)
print("RAM Address Verification Tool")
print("=" * 70)
print("\nThis tool helps verify RAM addresses from Data Crystal.")
print("Play the game and observe if values make sense:")
print("  - Kirby X/Y should change when moving")
print("  - Score should increase when collecting items")
print("  - Lives should decrease when dying")
print("  - Health should decrease when hit")
print("\nPress Ctrl+C to exit")
print("=" * 70)

input("\nPress Enter to start...")

pyboy = PyBoy(str(rom_path), window="SDL2")
pyboy.set_emulation_speed(1)

# Load init state if available
if state_path.exists():
    with open(state_path, "rb") as f:
        pyboy.load_state(f)
    print("✅ Loaded init state")

print("\n" + "=" * 70)
print("RAM Values (updates every second):")
print("=" * 70)

frame_count = 0
last_print = time.time()

try:
    while True:
        pyboy.tick()
        frame_count += 1

        # Print every second (60 frames)
        if time.time() - last_print >= 1.0:
            print(f"\n[Frame {frame_count}]")

            for name, addr in ADDRS.items():
                value = pyboy.memory[addr]
                print(f"  {name:12s}: 0x{value:02X} ({value:3d})")

            # Calculate score from BCD
            score = (pyboy.memory[0xD06F] * 100000 +
                    pyboy.memory[0xD070] * 10000 +
                    pyboy.memory[0xD071] * 1000 +
                    pyboy.memory[0xD072] * 100 +
                    pyboy.memory[0xD073] * 10)
            print(f"  {'Total Score':12s}: {score}")

            # Absolute X position
            abs_x = pyboy.memory[0xD053] + pyboy.memory[0xD05C]
            print(f"  {'Abs X Pos':12s}: {abs_x}")

            last_print = time.time()

except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("Verification Notes:")
    print("=" * 70)
    print("\n✅ Expected behaviors:")
    print("  - Kirby X: Changes (0-255) as Kirby moves horizontally")
    print("  - Kirby Y: Changes (0-255) as Kirby moves vertically")
    print("  - Scroll X: Increases as camera moves right")
    print("  - Score: BCD encoded, increases with items/enemies")
    print("  - Lives: Starts at 3, decreases when dying")
    print("  - Health: Decreases when hit, increases when collecting food")
    print("  - Game State: 0x01=normal, 0x06=dying/warpstar")
    print("\n❌ If any values seem wrong:")
    print("  - Document findings in docs/RAM_ADDRESSES.md")
    print("  - Search for correct address in PyBoy memory viewer")
    print("  - Update kirby_gym_env.py with correct addresses")
    print("=" * 70)

    pyboy.stop()
