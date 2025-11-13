#!/usr/bin/env python3
"""
verify_rom.py
Verifies that the Kirby's Dream Land ROM has the correct checksum.
"""
import hashlib
from pathlib import Path


def verify_rom():
    rom_path = Path("data/kirbys_dream_land.gb")
    # Expected values from Data Crystal (https://datacrystal.tcrf.net/wiki/Kirby%27s_Dream_Land)
    expected_size = 262144  # 256 KiB
    expected_rom_checksum = 0xADF9  # Game Boy ROM checksum from header
    expected_header_checksum = 0x98  # Game Boy header checksum

    print("=" * 60)
    print("Kirby's Dream Land ROM Verification")
    print("=" * 60)

    # Check if ROM exists
    if not rom_path.exists():
        print(f"❌ ROM not found: {rom_path}")
        print("\nExpected location: data/kirbys_dream_land.gb")
        print("Please copy the ROM file to the data/ directory.")
        return False

    print(f"✅ ROM found: {rom_path}")

    # Check file size
    file_size = rom_path.stat().st_size
    print(f"   File size: {file_size:,} bytes")

    if file_size != expected_size:
        print(f"⚠️  Size mismatch! Expected {expected_size:,} bytes")
        print("   This might not be the correct ROM version.")
    else:
        print(f"✅ File size matches: {expected_size:,} bytes")

    # Read ROM data
    with open(rom_path, 'rb') as f:
        rom_data = f.read()

    # Calculate MD5 checksum (for reference)
    md5 = hashlib.md5(rom_data).hexdigest()
    print(f"\nMD5 Checksum: {md5}")

    # Read Game Boy header checksums (official verification method)
    # Header checksum is at 0x014D (1 byte)
    # ROM checksum is at 0x014E-0x014F (2 bytes, big-endian)
    header_checksum = rom_data[0x014D]
    rom_checksum = (rom_data[0x014E] << 8) | rom_data[0x014F]

    print("\nGame Boy Header Verification:")
    print(f"   Header Checksum: 0x{header_checksum:02X} (expected: 0x{expected_header_checksum:02X})")
    print(f"   ROM Checksum:    0x{rom_checksum:04X} (expected: 0x{expected_rom_checksum:04X})")

    # Verify checksums
    header_ok = (header_checksum == expected_header_checksum)
    rom_ok = (rom_checksum == expected_rom_checksum)

    if header_ok and rom_ok:
        print("\n✅ All checksums match! ROM verified.")
        print("=" * 60)
        print("ROM is ready to use for training.")
        print("=" * 60)
        return True
    else:
        if not header_ok:
            print(f"\n⚠️  Header checksum mismatch!")
        if not rom_ok:
            print(f"\n⚠️  ROM checksum mismatch!")
        print("   This ROM file might not be the correct version.")
        print("   Expected: Kirby's Dream Land (U.S.A. & Europe)")
        print(f"   MD5 for reference: {md5}")
        return False


if __name__ == "__main__":
    success = verify_rom()
    exit(0 if success else 1)
