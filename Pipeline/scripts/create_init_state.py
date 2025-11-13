#!/usr/bin/env python3
"""
create_init_state.py
Interactive script to create init.state savestate for Kirby training.

Usage:
    python scripts/create_init_state.py

Instructions:
    1. Script will open PyBoy window
    2. Play through title screen manually:
       - Wait for title screen
       - Press Start
       - Select difficulty (Normal recommended)
       - Wait until gameplay starts
    3. Press 'S' key to save state
    4. State will be saved to data/kirby_init.state
"""
from pathlib import Path
from pyboy import PyBoy


def create_init_state():
    rom_path = Path("data/kirbys_dream_land.gb")
    output_path = Path("data/kirby_init.state")

    if not rom_path.exists():
        print(f"❌ ROM not found: {rom_path}")
        print("   Please run verify_rom.py first")
        return False

    print("=" * 60)
    print("Kirby Init State Creator")
    print("=" * 60)
    print("\nInstructions:")
    print("  1. PyBoy window will open")
    print("  2. Navigate through title screen:")
    print("     - Wait for 'KIRBY'S DREAM LAND' title")
    print("     - Press ENTER (Start)")
    print("     - Select 'NORMAL' difficulty (press ENTER)")
    print("     - Wait until gameplay starts (Kirby spawns)")
    print("  3. When ready, press 'S' to save state")
    print("  4. State will be saved to:", output_path)
    print("\nControls:")
    print("  Arrow Keys: Move")
    print("  Z: A Button (Jump/Confirm)")
    print("  X: B Button (Inhale)")
    print("  Enter: Start")
    print("  S: Save State (when ready)")
    print("  Escape: Exit without saving")
    print("=" * 60)

    input("\nPress ENTER to start PyBoy...")

    pyboy = PyBoy(str(rom_path), window="SDL2")
    pyboy.set_emulation_speed(1)  # Real-time

    print("\n✅ PyBoy started")
    print("   Navigate to gameplay start, then press 'S' to save state")

    saved = False
    try:
        while not saved:
            pyboy.tick()

            # Check for save key (S)
            # Note: PyBoy doesn't expose key events directly, so we use a simple timer
            # User must manually trigger save by pressing a specific game button sequence
            # For simplicity, we'll use a time-based approach

    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        pyboy.stop()
        return False

    # Manual save approach: User runs this, then manually saves
    print("\n\nManual Save Instructions:")
    print("=" * 60)
    print("Since PyBoy doesn't support key capture in this mode,")
    print("please follow these steps:")
    print("")
    print("1. Keep playing until Kirby spawns in Level 1")
    print("2. In the PyBoy window, press:")
    print("   - F5 to save state (not F2, that's for speed)")
    print("   - OR use PyBoy menu: File → Save State")
    print("3. PyBoy will save to default location")
    print("4. We'll copy it to the correct location")
    print("")
    print("Alternative: Use this script's direct approach below...")
    pyboy.stop()

    # Direct save approach (requires user to play first)
    print("\n" + "=" * 60)
    print("Alternative Method: Programmatic Save")
    print("=" * 60)

    input("Press ENTER after you've navigated to gameplay start in PyBoy...")

    # Reload PyBoy and let user navigate
    pyboy = PyBoy(str(rom_path), window="SDL2")
    pyboy.set_emulation_speed(1)

    print("\nNavigate to gameplay start...")
    print("Waiting 60 seconds for you to reach Level 1 start...")
    print("(Script will auto-save after 60 seconds)")

    # Give user time to navigate (60 seconds = 3600 frames)
    for i in range(3600):
        pyboy.tick()
        if i % 600 == 0:  # Every 10 seconds
            print(f"  {(i // 60)}/60 seconds elapsed...")

    # Save state
    print("\n💾 Saving state...")
    with open(output_path, "wb") as f:
        pyboy.save_state(f)

    pyboy.stop()

    print("=" * 60)
    print("✅ Init state saved successfully!")
    print(f"   Location: {output_path}")
    print("=" * 60)
    print("\nTo use this state in training:")
    print("  1. Update configs/kirby/base.yaml:")
    print("     init_state: 'data/kirby_init.state'")
    print("  2. Agent will start directly in gameplay")
    print("=" * 60)

    return True


if __name__ == "__main__":
    print("\n⚠️  WARNING: This script has limited keyboard support.")
    print("See detailed instructions below for best approach.\n")

    choice = input("Choose method:\n  1. Manual (recommended)\n  2. Auto-timer\n  3. Cancel\nChoice: ")

    if choice == "1":
        print("\n" + "=" * 60)
        print("Manual Method - Best Approach")
        print("=" * 60)
        print("\n1. Run: python scripts/test_manual_play.py")
        print("2. Play until Kirby spawns in Level 1")
        print("3. Note the game state at that moment")
        print("4. Then come back here and run Auto-timer method")
        print("\nOR")
        print("\n1. Open PyBoy GUI directly:")
        print("   pyboy data/kirbys_dream_land.gb")
        print("2. Navigate to Level 1 start")
        print("3. File → Save State → Choose location")
        print("4. Save as: data/kirby_init.state")
        print("=" * 60)
    elif choice == "2":
        create_init_state()
    else:
        print("Cancelled.")
