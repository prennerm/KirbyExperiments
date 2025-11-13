#!/usr/bin/env python3
"""
test_manual_play.py
Test script for playing Kirby's Dream Land manually with keyboard controls.
This verifies that the ROM loads correctly and PyBoy is working.
"""
from pathlib import Path
import sys


def test_manual_play():
    try:
        from pyboy import PyBoy
    except ImportError:
        print("❌ PyBoy not installed!")
        print("   Install with: pip install pyboy")
        return False

    rom_path = Path("data/kirbys_dream_land.gb")

    print("=" * 60)
    print("Kirby's Dream Land - Manual Play Test")
    print("=" * 60)

    # Check if ROM exists
    if not rom_path.exists():
        print(f"❌ ROM not found: {rom_path}")
        print("\nPlease run verify_rom.py first to check the ROM.")
        return False

    print(f"✅ ROM found: {rom_path}")
    print("\nLoading ROM with PyBoy...")

    try:
        pyboy = PyBoy(str(rom_path), window="SDL2")
        pyboy.set_emulation_speed(1)  # 1x real-time (60 FPS)
        print("✅ ROM loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load ROM: {e}")
        return False

    print("\n" + "=" * 60)
    print("Controls:")
    print("=" * 60)
    print("  Arrow Keys:  Move Kirby")
    print("  Z:          A Button (Jump/Confirm)")
    print("  X:          B Button (Inhale/Attack)")
    print("  Enter:      Start")
    print("  Backspace:  Select")
    print("  Ctrl+C:     Quit")
    print("=" * 60)
    print("\nWhat to test:")
    print("  1. Title screen appears")
    print("  2. Press Start to begin")
    print("  3. Kirby responds to arrow keys")
    print("  4. Jump works (Z)")
    print("  5. Inhale works (X)")
    print("=" * 60)
    print("\nStarting game... (press Ctrl+C to exit)\n")

    try:
        while True:
            pyboy.tick()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("✅ Manual play test completed!")
        print("=" * 60)
        print("\nIf you could see Kirby and control him, the setup is working!")
        print("Next steps:")
        print("  1. Find RAM addresses (see docs/KIRBY_SETUP.md)")
        print("  2. Implement kirby_gym_env.py")
        print("  3. Run training")
        pyboy.stop()
        return True
    except Exception as e:
        print(f"\n❌ Error during playback: {e}")
        pyboy.stop()
        return False


if __name__ == "__main__":
    success = test_manual_play()
    exit(0 if success else 1)
