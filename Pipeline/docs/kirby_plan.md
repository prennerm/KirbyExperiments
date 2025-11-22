# Kirby Pipeline – Integration Plan inspired by PyBoy-RL

## 1. Immediate Enhancements (short-term)

### 1.1 Reward & Game State Handling
- **Level progress**: adopt PyBoy-RL formula progress = scroll_x * 16 + (tile_offset) + kirby_x to obtain a continuous measure even when the camera scrolls.
- **Warpstar detection**: treat game_state == 6 as level completion; add a one-time reward and optional episode truncation/flag.
- **Boss rewards**: hook into oss_health deltas (damage + defeat) for sparse, meaningful bonuses.
- **Stillstand / backward penalties**: copy the heuristics (penalty for standing still, larger penalty for moving left when no boss active).
- **Death penalties**: align with PyBoy-RL levels (-1000 on death, -100 when low health and losing HP).

### 1.2 Kirby Environment (kirby_gym_env.py)
- Add helper functions _read_level_progress, _detect_warpstar, _boss_active using RAM addresses verified by PyBoy-RL.
- Restructure step() reward block to integrate the new signals (progress delta, boss hits, warpstar, death, idle).
- Include the additional stats (boss_hp, level_progress, game_state) in info so StatsCallback can log them.

### 1.3 Logging & Visualization
- Extend our StatsCallback to capture info["level_progress"], info["boss_hp"], info["game_state"].
- Update visualization scripts to plot level progress and boss health (similar to PyBoy-RL reward plots).
- Use the additional fields in reward plots to highlight warpstar events / boss transitions.
- **Neu (Nov 2025):** Der Action Space nutzt jetzt eine kuratierte Menge aus Einzel- und Zwei-Tasten-Kombinationen (z.B. Right, Jump, Right+Jump, B+Down, Start, Select), sodass der Agent realistisch mehrere Buttons gleichzeitig drücken kann.
- **Neu (Nov 2025, Update 2):** Alle PPO-Varianten schreiben ihre Trainingsmetriken (`train/clip_fraction`, `train/approx_kl`, `train/policy_loss`, `train/value_loss`, `train/entropy_loss`, `train/loss`, `train/explained_variance`) direkt in die `stats_*.csv`; LD-spezifische Felder (`training.ld.*`) kommen bei k_v3 zusätzlich dazu.

## 2. Mid-term Experiments

### 2.1 Action Space
- Evaluate PyBoy-RL approach (single + two-button combinations). Start with a curated subset (e.g. {Right, Jump, Jump+Right, Inhale+Right}) to avoid 36-action explosion.
- Compare with our current discrete actions; decide per variant (k_v1 vs k_v3/v4).

### 2.2 Boss-vs-Platform regimes
- Consider separate curriculum or reward weighting for boss fights vs platform sections (PyBoy-RL has distinct settings).
- Optionally implement a mode flag in env to detect when boss is active ? adapt reward/penalty scales.

### 2.3 Metric Logger Parallels
- Borrow ideas from MetricLogger.py (moving averages, reward/length/loss plots) and integrate into our Stats pipeline or add a lightweight post-processing tool to mirror their graphs.

## 3. Long-term / Optional

### 3.1 CustomPyBoyGym Evaluation
- PyBoy-RL uses a bespoke gym wrapper; we currently rely on our kirby_gym_env.py (Gymnasium). Staying on Gymnasium keeps compatibility with PPO pipelines; porting to their wrapper would require replacing the entire environment API. Recommendation: **do not switch**; instead cherry-pick features (reward logic, state extraction) while keeping our env interface intact.

### 3.2 DDQN vs PPO
- Their agent is DDQN-based (Pytorch). We are committed to PPO/RecurrentPPO(LD). Therefore, only borrow heuristics, not trainer code.

### 3.3 Advanced Visualization
- Once stats include level progress & boss info, create new visual notebooks akin to their README plots (reward vs boss phases, level completion timelines).

---

## Action Items Summary
| Priority | Task | Notes |
| -------- | ---- | ----- |
| ? High | Port level progress, warpstar detection, boss rewards into kirby_gym_env.py | immediate reward overhaul |
| ? High | Log new metrics (boss HP, level progress) & extend StatsCallback | required for visualization |
| ? High | Update visualization scripts to display new metrics | ensures feedback |
| ?? Medium | Experiment with two-button action combos (limited set) | compare performance |
| ?? Medium | Consider boss-mode reward tweaks (optional flag) | keep unified pipeline initially |
| ?? Medium | Add metric moving averages / reward plots like PyBoy-RL | integrate gradually |
| ? Low | Investigate CustomPyBoyGym for insights only (no direct adoption) | stick to Gymnasium |
| ? Low | Evaluate separate training regimes (boss-specific) later | after base pipeline stabilizes |
---

## Status Update – 2025-11-16

### Goals vs. Implementation
- **Reward + state plumbing (1.1/1.2)**: implemented in `kirby_gym_env.py`. Environment reports level progress via the PyBoy-RL formula, boss-health deltas, warpstar flags, and standstill/backtrack penalties. Manual play plus targeted savestate tests confirmed the rewards fire as designed; warpstar spikes seen in the tests mirror PyBoy-RL behaviour.
- **Stats logging (1.3)**: the new `kirby_pipeline.callbacks.StatsCallback` plus env-side `agent_stats` produce structured CSV/JSON rows containing `progress.*`, `boss.*`, and `status.*` fields (including warpstar/boss-active booleans). Output is stored under each run’s `logs/`.
- **Visualization prototype**: `scripts/plot_kirby_stats.py` loads stats CSVs and plots level progress (with warpstar scatter) plus boss HP timeline. Limitation: short smoke runs emit only a single warpstar datapoint (~step 2050), so the curve currently collapses into a vertical line—needs longer runs or smoothing.
- **Config-driven training**: `train.py` now reads all knobs (num_cpu, total_timesteps, run_label, etc.) from YAML. CLI stays minimal (`--variant`, `--config`, optional `--resume`). Added `configs/kirby/k_smoke.yaml` for quick 2k-step sanity runs without touching flagship configs.

### Current Pipeline Snapshot
- **Entrypoint**: `python -m kirby_pipeline.train --variant k_v1 --config configs/kirby/k_v1.yaml` (or `k_smoke` for short tests). Resolved configs get archived beside logs/checkpoints.
- **Artifacts produced**: `logs/stats_*.csv` with the new metrics, checkpoints every `save_freq`, TensorBoard traces, and optional plots via `plot_kirby_stats.py`.
- **Open issues**: reward plots need better temporal coverage/smoothing; documentation for manual testing/plot usage should be expanded; action-space/boss-mode experiments remain TODO.


### Status Update – 2025-11-22
- **Training Metrics Logging:** Die Modelle PPOWithMetrics, RecurrentPPOWithMetrics und RecurrentPPOLD füllen `latest_train_metrics` mit allen TensorBoard-Metriken (clip_fraction, approx_kl, policy/value/entropy loss, total loss, explained_variance). Die Stats-CSV enthält dadurch einheitliche `train/*`-Spalten für k_v1–k_v3; `scripts/analyze_run_stats.py` zeigt keine `n/a` mehr.
