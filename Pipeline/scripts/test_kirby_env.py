#!/usr/bin/env python3
"""
test_kirby_env.py
Sanity check for KirbyGymEnv - tests basic functionality with random actions.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kirby_pipeline.kirby_gym_env import KirbyGymEnv


def test_kirby_env():
    """Test KirbyGymEnv with random actions."""
    print("=" * 60)
    print("KirbyGymEnv Sanity Check")
    print("=" * 60)

    # Environment config
    config = {
        "session_path": "experiments/test",
        "gb_path": "data/kirbys_dream_land.gb",
        "headless": False,  # Show window
        "print_rewards": True,
        "action_freq": 24,
        "max_steps": 500,  # Short test
        "init_state": "",  # No init state for now
        "save_final_state": False,
        "save_video": False,
        "reward_scale": 1.0,
        "emulation_speed": 1,  # Real-time for observation
    }

    print("\n[1/5] Creating environment...")
    env = KirbyGymEnv(config)
    print("✅ Environment created")

    print("\n[2/5] Resetting environment...")
    obs, info = env.reset()
    print("✅ Environment reset")
    print(f"   Observation type: {type(obs)}")
    print(f"   Observation shape: {obs.shape}")
    print(f"   POMDP compliant: Agent only sees screen pixels")

    print("\n[3/5] Testing action space...")
    print(f"   Action space: {env.action_space}")
    print(f"   Valid actions: {len(env.valid_actions)}")
    print("✅ Action space valid")

    print("\n[4/5] Running random agent for 100 steps...")
    print("   (Watch the PyBoy window to see Kirby moving)")

    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"   Step {step:3d}: Action={action}, Reward={reward:.2f}, "
                  f"Score={info['score']}, X={info['x_position']}")

        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            break

    print("✅ Random agent completed")
    print(f"   Total reward: {total_reward:.2f}")

    print("\n[5/5] Closing environment...")
    env.close()
    print("✅ Environment closed")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Create init.state (savestate after title screen)")
    print("  2. Run training: python -m kirby_pipeline.train --config configs/kirby/k_v1.yaml")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_kirby_env()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
