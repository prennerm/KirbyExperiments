"""
kirby_gym_env.py
Gymnasium environment for Kirby's Dream Land (Game Boy) using PyBoy.

Based on Pokemon Red environment but simplified for Kirby's linear level structure.
RAM addresses sourced from: https://datacrystal.tcrf.net/wiki/Kirby%27s_Dream_Land/RAM_map
"""
import uuid
from pathlib import Path
from itertools import permutations
import numpy as np
from skimage.transform import downscale_local_mean
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent


class KirbyGymEnv(Env):
    """
    Gymnasium environment for training RL agents on Kirby's Dream Land.

    Observation Space:
        - screens: Downscaled game screen (72x80x3 with frame stacking)
        - health: Kirby's current health (normalized 0-1)
        - lives: Number of lives (normalized 0-1)
        - score: Current score (normalized)

    Action Space:
        Discrete(N): curated set of single buttons and two-button combos

    Reward Components:
        - score_delta: Points gained (encourages item collection, enemy defeat)
        - x_progress: Moving right (primary goal in linear levels)
        - level_complete: Finishing a level (sparse, large reward)
        - death_penalty: Dying (negative, moderate)
        - time_penalty: Efficiency (small constant negative)
    """

    # RAM addresses from Data Crystal
    RAM_KIRBY_X = 0xD05C
    RAM_KIRBY_Y = 0xD05D
    RAM_HEALTH = 0xD086
    RAM_LIVES = 0xD089
    RAM_SCORE_100K = 0xD06F
    RAM_SCORE_10K = 0xD070
    RAM_SCORE_1K = 0xD071
    RAM_SCORE_100 = 0xD072
    RAM_SCORE_10 = 0xD073
    RAM_GAME_STATE = 0xD02C
    RAM_BOSS_HEALTH = 0xD093
    RAM_SCROLL_X = 0xD053
    WARPSTAR_STATE = 0x06

    def __init__(self, config=None):
        super().__init__()

        # Config parameters
        self.s_path = Path(config["session_path"])
        self.headless = config.get("headless", True)
        self.print_rewards = config.get("print_rewards", False)
        self.act_freq = config.get("action_freq", 24)  # Frames per action
        self.max_steps = config.get("max_steps", 10000)
        self.init_state = config.get("init_state", None)
        self.save_final_state = config.get("save_final_state", True)
        self.save_video = config.get("save_video", False)

        # Instance tracking
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])
        self.worker_rank = config.get("worker_rank", 0)
        self.reset_count = 0

        # Frame stacking for temporal info (v1/v2 style)
        self.frame_stacks = 3

        # Reward weights
        self.reward_scale = config.get("reward_scale", 1.0)
        self.score_weight = config.get("score_weight", 0.001)
        self.progress_weight = config.get("progress_weight", 0.02)
        self.level_complete_reward = config.get("warpstar_reward", 20)
        self.warpstar_reset_progress = config.get("warpstar_reset_progress", 200)
        self.boss_hit_reward = config.get("boss_hit_reward", 10)
        self.boss_defeat_reward = config.get("boss_defeat_reward", 50)
        self.standstill_penalty = config.get("standstill_penalty", 0.05)
        self.backtrack_penalty = config.get("backtrack_penalty", 0.1)
        self.edge_reward = config.get("edge_reward", 1.0)
        self.death_penalty = config.get("death_penalty", 100)
        self.time_penalty = config.get("time_penalty", 0.001)
        self.reward_clip = config.get("reward_clip", 5.0)

        # Action space: singles + selected two-button combos
        combo_sources = [
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        single_only = [
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT,
        ]

        self.press_to_release = {
            WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
        }

        action_combos = [(btn,) for btn in combo_sources]
        action_combos.extend(permutations(combo_sources, 2))
        action_combos.extend((btn,) for btn in single_only)

        self.valid_actions = action_combos
        self.release_actions = [
            tuple(self.press_to_release[event] for event in reversed(combo))
            for combo in action_combos
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Observation space - POMDP: Only screens visible to agent
        # Agent must learn health/lives/score from visual information
        self.output_shape = (72, 80, self.frame_stacks)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.output_shape, dtype=np.uint8
        )

        # Initialize PyBoy
        self.s_path.mkdir(parents=True, exist_ok=True)
        rom_path = config.get("gb_path", "data/kirbys_dream_land.gb")
        head = "null" if self.headless else "SDL2"

        self.pyboy = PyBoy(
            rom_path,
            window=head,
            sound_emulated=False,
        )

        if not self.headless:
            self.pyboy.set_emulation_speed(config.get("emulation_speed", 6))

        # State tracking
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros(self.frame_stacks, dtype=np.uint8)

        # Episode stats
        self.step_count = 0
        self.episode_reward = 0
        self.agent_stats = []

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Load initial state if provided
        if self.init_state and Path(self.init_state).exists():
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
        else:
            reset_fn = getattr(self.pyboy, "reset_game", None) or getattr(self.pyboy, "reset", None)
            if callable(reset_fn):
                reset_fn()
            else:
                for _ in range(300):
                    self.pyboy.tick()

        # Reset tracking variables
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros(self.frame_stacks, dtype=np.uint8)
        self.step_count = 0
        self.episode_reward = 0
        self.agent_stats = []
        self.reset_count += 1

        # Read initial game state
        self.prev_score = self._read_score()
        self.prev_x = self._read_x_position()
        self.prev_lives = self._read_lives()
        self.max_x_position = self.prev_x
        self.prev_level_progress = self._read_level_progress()
        self.prev_boss_health = self._read_boss_health()
        self.prev_game_state = self._read_game_state()
        self.warpstar_locked = False

        return self._get_obs(), {}

    def step(self, action):
        """Execute action and return observation, reward, done, info."""
        # Execute action for act_freq frames
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

        # Press buttons for selected action (can be single or combo)
        for press_event in self.valid_actions[action]:
            self.pyboy.send_input(press_event)

        for i in range(self.act_freq):
            # Release button after first frame (realistic input timing)
            if i == 8:
                for release_event in self.release_actions[action]:
                    self.pyboy.send_input(release_event)

            self.pyboy.tick()

        # Update screen buffer
        self._update_screen_buffer()

        # Read current game state
        current_score = self._read_score()
        current_x = self._read_x_position()
        current_lives = self._read_lives()
        current_health = self._read_health()
        current_progress = self._read_level_progress()
        current_boss_health = self._read_boss_health()
        current_game_state = self._read_game_state()
        boss_active = current_boss_health > 0

        # Calculate reward components
        score_delta = current_score - self.prev_score
        x_progress = max(0, current_x - self.max_x_position)
        progress_delta = current_progress - self.prev_level_progress
        boss_health_delta = self.prev_boss_health - current_boss_health
        died = (current_lives < self.prev_lives)

        # Update max position
        if current_x > self.max_x_position:
            self.max_x_position = current_x

        # Reward calculation
        reward = 0
        reward += score_delta * self.score_weight
        reward += max(0, progress_delta) * self.progress_weight
        reward -= self.time_penalty

        if progress_delta <= 0 and not boss_active:
            reward -= self.standstill_penalty

        if progress_delta < 0 and not boss_active:
            reward -= self.backtrack_penalty

        if current_boss_health < self.prev_boss_health:
            reward += self.boss_hit_reward

        if current_boss_health == 0 and self.prev_boss_health > 0:
            reward += self.boss_defeat_reward

        if self.warpstar_locked:
            if current_game_state != self.WARPSTAR_STATE and current_progress < self.warpstar_reset_progress:
                self.warpstar_locked = False

        if (
            current_game_state == self.WARPSTAR_STATE
            and self.prev_game_state != self.WARPSTAR_STATE
            and not self.warpstar_locked
        ):
            reward += self.level_complete_reward
            self.warpstar_locked = True

        if died:
            reward -= self.death_penalty

        if not boss_active:
            if current_x == 76:
                reward += self.edge_reward
            elif current_x == 68:
                reward -= self.edge_reward

        # Episode termination conditions
        self.step_count += 1
        terminated = False
        truncated = False

        # Check if died and out of lives
        if current_lives == 0:
            terminated = True
            reward -= 500  # Extra penalty for game over

        # Check max steps
        if self.step_count >= self.max_steps:
            truncated = True

        # Update previous state
        self.prev_score = current_score
        self.prev_x = current_x
        self.prev_lives = current_lives
        self.prev_level_progress = current_progress
        self.prev_boss_health = current_boss_health
        self.prev_game_state = current_game_state

        # Apply reward scale
        reward *= self.reward_scale
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        self.episode_reward += reward

        # Print rewards if enabled
        if self.print_rewards and self.step_count % 100 == 0:
            print(f"[Worker {self.worker_rank}] Step {self.step_count}: "
                  f"Score={current_score}, X={current_x}, Lives={current_lives}, "
                  f"Reward={reward:.2f}, Total={self.episode_reward:.2f}")

        info = {
            "score": current_score,
            "score_delta": score_delta,
            "x_position": current_x,
            "level_progress": current_progress,
            "progress_delta": progress_delta,
            "lives": current_lives,
            "health": current_health,
            "boss_health": current_boss_health,
            "boss_health_delta": boss_health_delta,
            "game_state": current_game_state,
            "warpstar": current_game_state == self.WARPSTAR_STATE,
            "boss_active": boss_active,
            "died": died,
            "episode_reward": self.episode_reward,
            "worker": self.worker_rank,
            "step_count": self.step_count,
        }
        self._append_agent_stats(action, reward, info)

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Render is handled by PyBoy window (if not headless)."""
        pass

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'pyboy'):
            self.pyboy.stop()

    def _append_agent_stats(self, action, reward, info):
        """Collect per-step stats for offline logging."""
        stat = {
            "step": info.get("step_count", self.step_count),
            "last_action": int(action),
            "reward_step": float(reward),
            "reward_total": float(self.episode_reward),
            "score": int(info.get("score", 0)),
            "score_delta": int(info.get("score_delta", 0)),
            "x_position": int(info.get("x_position", 0)),
            "level_progress": int(info.get("level_progress", 0)),
            "level_progress_delta": int(info.get("progress_delta", 0)),
            "boss_health": int(info.get("boss_health", 0)),
            "boss_health_delta": int(info.get("boss_health_delta", 0)),
            "game_state": int(info.get("game_state", 0)),
            "warpstar": bool(info.get("warpstar", False)),
            "boss_active": bool(info.get("boss_active", False)),
            "died": bool(info.get("died", False)),
            "lives": int(info.get("lives", 0)),
            "health": int(info.get("health", 0)),
            "worker": int(info.get("worker", self.worker_rank)),
        }
        self.agent_stats.append(stat)

    # ========================================================================
    # RAM Reading Helpers
    # ========================================================================

    def _read_score(self):
        """Read score from BCD-encoded memory (5 bytes)."""
        score = 0
        score += self.pyboy.memory[self.RAM_SCORE_100K] * 100000
        score += self.pyboy.memory[self.RAM_SCORE_10K] * 10000
        score += self.pyboy.memory[self.RAM_SCORE_1K] * 1000
        score += self.pyboy.memory[self.RAM_SCORE_100] * 100
        score += self.pyboy.memory[self.RAM_SCORE_10] * 10
        return score

    def _read_x_position(self):
        """Read Kirby's absolute X position (screen-relative + scroll)."""
        kirby_x = self.pyboy.memory[self.RAM_KIRBY_X]
        scroll_x = self.pyboy.memory[self.RAM_SCROLL_X]
        return scroll_x + kirby_x

    def _read_health(self):
        """Read Kirby's health (normalized to 0-1)."""
        health = self.pyboy.memory[self.RAM_HEALTH]
        return health / 255.0  # Normalize

    def _read_lives(self):
        """Read number of lives."""
        return self.pyboy.memory[self.RAM_LIVES]

    def _read_game_state(self):
        """Read game state byte (0x01=normal, 0x06=dying)."""
        return self.pyboy.memory[self.RAM_GAME_STATE]

    def _read_boss_health(self):
        """Read boss HP."""
        return self.pyboy.memory[self.RAM_BOSS_HEALTH]

    def _read_level_progress(self):
        """Approximate absolute progress across the level."""
        scroll_x = self.pyboy.memory[self.RAM_SCROLL_X]
        kirby_x = self.pyboy.memory[self.RAM_KIRBY_X]
        fine_offset = 0
        manager = getattr(self.pyboy, "botsupport_manager", None)
        if callable(manager):
            try:
                screen = manager().screen()
                tilemap = screen.tilemap_position_list()
                scx = tilemap[16][0]
                fine_offset = (scx - 7) % 16
            except Exception:
                fine_offset = 0
        return scroll_x * 16 + fine_offset + kirby_x

    # ========================================================================
    # Observation Helpers
    # ========================================================================

    def _get_obs(self):
        """
        Construct observation - POMDP compliant.

        Agent only sees screen pixels. Health, lives, and score are visible
        in the game's HUD, so agent must learn to read them from pixels.
        """
        return self.recent_screens.copy()

    def _update_screen_buffer(self):
        """Capture current screen and update frame stack."""
        # Get raw screen (160x144 grayscale)
        screen = self.pyboy.screen.ndarray[:, :, 0]  # Take first channel (grayscale)

        # Downscale to 72x80 (match Pokemon environment)
        # Original: 160x144 -> Target: 72x80
        # Scale factors: 144/72 = 2.0, 160/80 = 2.0
        downscaled = downscale_local_mean(screen, (2, 2)).astype(np.uint8)

        # Roll frame stack and add new frame
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = downscaled
