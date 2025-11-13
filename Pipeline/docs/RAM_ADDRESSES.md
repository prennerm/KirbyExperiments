# Kirby's Dream Land - RAM Addresses

Source: [Data Crystal - Kirby's Dream Land RAM Map](https://datacrystal.tcrf.net/wiki/Kirby%27s_Dream_Land/RAM_map)

## Core Game State

### Position & Movement

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xD05C** | 1 byte | Kirby X position (screen-relative) | Max $4C at screen edge |
| **0xD05D** | 1 byte | Kirby Y position (screen-relative) | Relative to screen |
| **0xD054** | 1 byte | Kirby X sub-pixels | For precise movement |
| **0xD056** | 1 byte | Kirby Y sub-pixels | For precise movement |
| **0xD053** | 1 byte | Visual Scroll X | Camera position |
| **0xD055** | 1 byte | Visual Scroll Y | Camera position |

### Velocity

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xD074-0xD075** | 2 bytes | Kirby X Speed | Pixels + sub-pixels |
| **0xD076-0xD077** | 2 bytes | Kirby X Speed Max | 1.2 px/frame normal, 0.75 flying, 0.66 underwater |
| **0xD078-0xD079** | 2 bytes | Kirby Y Speed | Pixels + sub-pixels |
| **0xD07A-0xD07B** | 2 bytes | Kirby Y Speed Max | 2.0 px/frame normal, 0.8 flying |

### Stats & Progression

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xD086** | 1 byte | Kirby Health | Current HP |
| **0xD089** | 1 byte | Lives | Number of lives |
| **0xD06F-0xD073** | 5 bytes | Score Display | BCD: 100000s, 10000s, 1000s, 100s, 10s |
| **0xD08B-0xD08D** | 3 bytes | Score / 10 | Capped at $01869F (99999), displays 999990 |
| **0xD093** | 1 byte | Boss Health | Current boss HP |

### Game State

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xD02C** | 1 byte | Game State | $01=normal, $05=drinking bottle, $06=warpstar/dying |
| **0xD066** | 1 byte | Inhale Timer | Increments while inhaling, freezes when stopped |

### Graphics

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xD080** | 1 byte | Background Brightness | Copied to BGP ($FF47) |
| **0xD081** | 1 byte | Sprite Brightness | Contrast control |

### Sprites (OAM)

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xC000-0xC09F** | 160 bytes | Sprite Info | 4 bytes each: y, x, tile#, attrs. Copied to OAM ($FE00-$FE9F) |

### Level Data

| Address | Size | Function | Notes |
|---------|------|----------|-------|
| **0xCA00-0xCAA3** | 164 bytes | Block Solidity | $00=air, $01=platform, $02=solid, $06=water, $07=spikes |
| **0xCB00-0xCD94** | 660 bytes | Level Tilemap Updates | 6 bytes per 2x2 tile block |

## Priority Addresses for RL Agent

### Essential (must read every step):
- **0xD05C**: Kirby X position
- **0xD05D**: Kirby Y position
- **0xD086**: Health
- **0xD089**: Lives
- **0xD06F-0xD073**: Score (5 bytes)

### Important (for reward shaping):
- **0xD02C**: Game state (detect death: $06)
- **0xD093**: Boss health (level completion)
- **0xD053**: Scroll X (level progress)

### Optional (for advanced features):
- **0xD066**: Inhale timer (encourage using abilities)
- **0xD074-0xD075**: X velocity (movement analysis)
- **0xCA00-0xCAA3**: Block solidity (collision prediction)

## Implementation Notes

### Reading Score
Score is stored in **BCD (Binary Coded Decimal)**:
- 0xD06F: 100,000s place
- 0xD070: 10,000s place
- 0xD071: 1,000s place
- 0xD072: 100s place
- 0xD073: 10s place

**Example:** Score of 12,340 points:
```
0xD06F = 0x00  (0 * 100,000)
0xD070 = 0x01  (1 * 10,000)
0xD071 = 0x02  (2 * 1,000)
0xD072 = 0x03  (3 * 100)
0xD073 = 0x04  (4 * 10)
Total: 12,340
```

Python code:
```python
def read_score(pyboy):
    score = 0
    score += pyboy.memory[0xD06F] * 100000
    score += pyboy.memory[0xD070] * 10000
    score += pyboy.memory[0xD071] * 1000
    score += pyboy.memory[0xD072] * 100
    score += pyboy.memory[0xD073] * 10
    return score
```

### Detecting Level Progress
Use **0xD053 (Scroll X)** to track horizontal progress. Higher values = further right.

Combine with **0xD05C (Kirby X)** for absolute world position:
```python
absolute_x = pyboy.memory[0xD053] + pyboy.memory[0xD05C]
```

### Detecting Death
Check **0xD02C (Game State)**:
- Normal gameplay: 0x01
- Dying/Warpstar: 0x06

Also check **lives delta**: if lives decreased, Kirby died.

### Detecting Level Completion
Watch for:
1. **Boss health (0xD093)** reaching 0
2. **Game state (0xD02C)** changing to victory sequence
3. **Score increase** (level completion bonus)

## ROM Information

From Data Crystal:

| Property | Value |
|----------|-------|
| **Internal Name** | KIRBY DREAM LAND |
| **Region** | U.S.A. & Europe |
| **Type** | Grayscale Game |
| **SGB Support** | No |
| **Cartridge Type** | ROM + MBC1 |
| **ROM Size** | 256 KiB (262,144 bytes) |
| **ROM Checksum** | 0xADF9 (at header 0x014E-0x014F) |
| **Header Checksum** | 0x98 (at header 0x014D) |
| **SRAM Size** | 0 KiB (no save data) |

**Note:** Use `verify_rom.py` script to check your ROM against these official checksums.

## Next Steps

1. ✅ RAM addresses documented (this file)
2. ⏳ Verify ROM checksum matches
3. ⏳ Test reading RAM addresses manually with PyBoy
4. ⏳ Implement `kirby_gym_env.py` using these addresses
5. ⏳ Test reward calculation with real gameplay data
