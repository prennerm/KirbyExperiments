"""
Lambda-Discrepancy extensions for the Kirby PPO pipeline.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy
from stable_baselines3.common.utils import explained_variance

from .training_metrics import build_training_metrics_snapshot


class CnnLstmPolicyLD(CnnLstmPolicy):
    """Recurrent CNN policy with an auxiliary lambda-discrepancy value head."""

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        latent_dim = self.mlp_extractor.latent_dim_vf
        self.value_net_ld = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        for module in self.value_net_ld.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        """
        Returns (main critic values, lambda-discrepancy values, log_prob, entropy).
        Mirrors the RecurrentPPO evaluate_actions flow while tapping the critic features
        for the auxiliary head.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_feat, vf_feat = features, features
        else:
            pi_feat, vf_feat = features

        latent_pi, lstm_states_pi = self._process_sequence(
            pi_feat, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_feat, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (
                lstm_states_pi[0].detach(),
                lstm_states_pi[1].detach(),
            )
        else:
            latent_vf = self.critic(vf_feat)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        mc_values = self.value_net(latent_vf)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        ld_values = self.value_net_ld(latent_vf)
        return mc_values, ld_values, log_prob, entropy


class RecurrentPPOLD(RecurrentPPO):
    """Recurrent PPO variant with lambda-discrepancy auxiliary loss."""

    def __init__(self, *args, ld_coef: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ld_coef = ld_coef
        self.latest_train_metrics = {}

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, pg_losses, mc_losses, ld_losses = [], [], [], []
        clip_fractions, approx_kl_divs, total_losses = [], [], []
        continue_training = True
        epochs_done = 0

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                mc_values, ld_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                mc_values = mc_values.flatten()
                ld_values = ld_values.flatten()
                log_prob = log_prob.flatten()

                old_values = rollout_data.old_values.flatten()
                returns = rollout_data.returns.flatten()
                episode_starts = rollout_data.episode_starts.float().flatten()
                old_log_prob = rollout_data.old_log_prob.flatten()

                not_start = 1.0 - episode_starts
                next_not_start = th.cat(
                    [not_start[1:], th.zeros(1, device=not_start.device)],
                    dim=0,
                )
                next_returns = th.cat(
                    [returns[1:], th.zeros(1, device=returns.device)],
                    dim=0,
                )
                rewards = returns - self.gamma * next_returns * next_not_start
                next_values = th.cat(
                    [old_values[1:], th.zeros(1, device=old_values.device)],
                    dim=0,
                )
                td0_targets = rewards + self.gamma * next_values * next_not_start
                ld_targets = (td0_targets - old_values).detach()

                advantages = rollout_data.advantages.flatten()
                ratio = th.exp(log_prob - old_log_prob)
                policy_loss = -th.mean(
                    th.min(
                        advantages * ratio,
                        advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                    )
                )
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                log_ratio = log_prob - old_log_prob
                approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)).item()
                approx_kl_divs.append(approx_kl_div)

                mc_loss = F.mse_loss(mc_values, returns)
                ld_loss = F.l1_loss(ld_values, ld_targets)
                ent_loss = -th.mean(entropy)

                loss = (
                    policy_loss
                    + self.ent_coef * ent_loss
                    + self.vf_coef * mc_loss
                    + self.ld_coef * ld_loss
                )

                ld_term = ld_loss.item()
                ld_component = self.ld_coef * ld_term
                total_loss = loss.item()
                total_losses.append(total_loss)
                ld_ratio = ld_component / (abs(total_loss) + 1e-8)
                policy_value = float(policy_loss.item())
                mc_value = float(mc_loss.item())
                ld_value = float(ld_loss.item())
                entropy_value = float(ent_loss.item())
                extras = {
                    "ld": {
                        "term": float(ld_term),
                        "component": float(ld_component),
                        "ratio": float(ld_ratio),
                    },
                    "ld_loss": ld_value,
                    "train/ld_loss": ld_value,
                    "train/ld_ratio": float(ld_ratio),
                }
                train_snapshot = build_training_metrics_snapshot(
                    policy_loss=policy_value,
                    value_loss=mc_value,
                    entropy_loss=entropy_value,
                    approx_kl=float(approx_kl_div),
                    clip_fraction=float(clip_fraction),
                    total_loss=float(total_loss),
                    extras=extras,
                )
                self.latest_train_metrics = train_snapshot

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                mc_losses.append(mc_loss.item())
                ld_losses.append(ld_loss.item())
                entropy_losses.append(ent_loss.item())

            epochs_done += 1
            if not continue_training:
                break

        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/mc_loss", np.mean(mc_losses))
        self.logger.record("train/ld_loss", np.mean(ld_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        if clip_fractions:
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if approx_kl_divs:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        if total_losses:
            self.logger.record("train/loss", np.mean(total_losses))

        if ld_losses:
            with th.no_grad():
                avg_ld = th.abs(ld_targets).mean().item()
                td0_mean = td0_targets.mean().item()
                mc_mean = returns.mean().item()
                self.logger.record("train/ld_target_abs_mean", avg_ld)
                self.logger.record("train/bootstrap_value_mean", td0_mean)
                self.logger.record("train/mc_value_mean", mc_mean)
                self.logger.record(
                    "train/bootstrap_vs_mc_ratio", td0_mean / (mc_mean + 1e-8)
                )
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/explained_variance", explained_var)
        if self.latest_train_metrics:
            self.latest_train_metrics["explained_variance"] = float(explained_var)
            self.latest_train_metrics["train/explained_variance"] = float(explained_var)

        self._n_updates += epochs_done


__all__ = ["CnnLstmPolicyLD", "RecurrentPPOLD"]
