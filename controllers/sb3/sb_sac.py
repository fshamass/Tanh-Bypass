"""
SACBypass — single-phase SAC with the masked gradient-direction bypass.

This is ORDINARY SAC — critic, actor and entropy temperature all co-trained, nothing
frozen — with ONE extra term added to the actor loss: the critic's action-gradient
delivered to the policy mean WITHOUT the tanh throttle (1 - a^2).

Why (full story in co_trained_bypass_plan.md)
---------------------------------------------
A tanh-squashed Gaussian policy has da/du = 1 - a^2, so the critic's advice reaches the
pre-tanh mean u attenuated to ~0 near the action bounds. The policy mean then under-shoots
optimal near-bound actions (e.g. it stops short of full brake). The bypass adds a term
whose gradient is the un-throttled critic gradient, restoring what tanh removed.

Actor loss (per batch), summed over action dims d, averaged over the batch — the bypass
uses the SAME divisor as the standard term, which is what keeps their ratio pure geometry:

    L_actor = mean_i[ alpha * log pi(a_i|s_i) - Q(s_i, a_i) ]      (standard SAC)
              - mean_i[ sum_d  m_id * w_id * u_id ]                (bypass, when on)

    u    = pre-tanh mean head (deterministic; NOT the sampled action)
    a_mu = tanh(u)
    w_d  = dQ_min/da_d evaluated at a_mu, DETACHED (magnitude kept, not just sign)
    m_d  = 1[blamed_d] * 1[a_on <= |a_mu_d| < 1 - delta_wall]      (per-dim mask)

The bypass term is linear in u with w and m detached, so its gradient on u_id is exactly
-m_id * w_id / B; on a surviving near-saturated dim it hands the actor the un-throttled
critic gradient the standard term loses to (1 - a^2), and the ratio between them is
1/(1 - a^2) — geometry, not a knob. See eq. (37) in the plan.

Design choices
--------------
- MASKED, per action dimension (see `_bypass_mask`). Two conjoined factors: the stored
  per-dim blame (recorded under the OLD policy, at collection time) and a live saturation
  band on the CURRENT policy's mean. The band is what keeps stale blame honest — a dim
  that was blamed but has since left saturation no longer gets pushed, and a dim already
  at the wall is not pushed further.
  NOTE: this is a GATE, and the plan (§2.3, §7.6) retired gating after two candidates
  failed in the toy study. It is a different gate from the retired wall-zone/wall-ward
  mask, but the comparison it supports is "blame+band-gated bypass vs baseline", not the
  ungated bypass the toy validated. Set `a_on=0.0`, `delta_wall=0.0` and blame everywhere
  to recover the ungated behaviour.
- `bypass_off=True` drops the term entirely, giving a plain-SAC baseline that is otherwise
  identical (same collection, same batches, same critic, same entropy) — so the two arms
  differ in exactly one loss term.
- CRITIC NOT FROZEN. It keeps learning its own Bellman target, which is what keeps w
  honest as the policy moves. This is the key departure from the retired two-phase
  frozen-critic design, whose stale critic degraded the policy over long runs.
- Entropy stays on (auto-tuned), exactly as standard SAC. The bypass is added alongside
  the entropy-regularized actor loss, not in place of it.

Loss VALUE of the bypass is meaningless (linear in u, unbounded below); read its gradient
effect through the mean-action / saturation diagnostics, never as a training curve.
"""

from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class SACBypass(SAC):
    """SAC + optional ungated gradient-bypass term on the actor loss. Critic co-trained.

    Only `__init__` (to accept `bypass_off`) and `train` (to add the bypass term and its
    diagnostics) differ from stable_baselines3.SAC. Everything else — collection,
    exploration, replay buffer, target network, entropy tuning, save/load — is stock SB3.
    """

    def __init__(
        self,
        *args: Any,
        bypass_off: bool = False,
        entropy_off: bool = False,
        entropy_frozen: bool = False,
        a_on: float = 0.9,
        delta_wall: float = 0.01,
        bypass_dims: Any = None,
        **kwargs: Any,
    ):
        # Stored before super().__init__ so they survive even if _setup_model runs early.
        self.bypass_off = bool(bypass_off)
        self.entropy_off = bool(entropy_off)   # True -> alpha=0, SAC without the entropy term
        self.entropy_frozen = bool(entropy_frozen)  # True -> alpha held at the loaded value (no tuning)
        self.a_on = float(a_on)                # band floor: below this, tanh is not throttling
        self.delta_wall = float(delta_wall)    # band ceiling is 1 - delta_wall (already at bound)
        self.bypass_dims = bypass_dims         # None = every action dim (see the property)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    @property
    def bypass_dims(self):
        """Action dims the bypass may act on; None means all of them.

        Set to e.g. [1] to run throttle-only and leave steering as plain SAC — the ablation
        for 'is the out-of-road rise coming from amplified steering?'.
        """
        return self._bypass_dims

    @bypass_dims.setter
    def bypass_dims(self, dims) -> None:
        self._bypass_dims = None if dims is None else tuple(int(d) for d in dims)
        self._dim_selector = None              # rebuilt lazily; never stale

    def _dim_names(self):
        """Log-friendly names per action dim. MetaDrive: [steering, throttle_brake]."""
        n = int(np.prod(self.action_space.shape))
        known = ("steer", "throttle")
        return [known[d] if n == len(known) else f"dim{d}" for d in range(n)]

    def _dim_mask(self, like: th.Tensor) -> th.Tensor:
        """(1, A) row of 0/1 selecting the dims the bypass is allowed to touch."""
        if self._dim_selector is None or self._dim_selector.shape[1] != like.shape[1]:
            sel = th.zeros(1, like.shape[1], device=like.device, dtype=like.dtype)
            if self._bypass_dims is None:
                sel += 1.0
            else:
                for d in self._bypass_dims:
                    sel[0, d] = 1.0
            self._dim_selector = sel
        return self._dim_selector

    # ------------------------------------------------------------------
    # The per-dim mask
    # ------------------------------------------------------------------
    def _bypass_mask(self, a_mu: th.Tensor, blame: th.Tensor) -> th.Tensor:
        """m_d = 1[blamed_d] * 1[a_on <= |a_mu_d| <= 1 - delta_wall].  Shape (B, A).

        The band is the FLAT SHOULDER of tanh, on either side. Taking |a_mu| folds the two
        sign branches into one test — it is equivalent to

            a_on <= a_mu_d <= 1 - delta_wall     (positive shoulder)
         or -a_on >= a_mu_d >= -1 + delta_wall   (negative shoulder)

        and excludes the linear middle as well as the very tip of each shoulder.

        Both factors are per action DIMENSION, and they ask different questions:

        blamed_d   — recorded at collection time, under the OLD policy: did this dim
                     precede a failure? Stored per-dim in CustomReplayBuffer.blame.
        band       — evaluated live, under the CURRENT policy: is the mean action where
                     the bypass has anything to do? Inside `a_on` tanh is barely throttling
                     and the standard term already delivers the gradient; past
                     `1 - delta_wall` the mean is effectively at the bound already, so
                     pushing further only inflates |u| without changing the action.

        The band re-checks the stale blame against where the policy sits NOW: a dim that
        was blamed but has since moved off the shoulder is dropped.
        """
        blamed = (blame > 0.0).float()
        abs_a = a_mu.abs()
        band = ((abs_a >= self.a_on) & (abs_a <= 1.0 - self.delta_wall)).float()
        return blamed * band * self._dim_mask(a_mu)

    # ------------------------------------------------------------------
    # The un-throttled critic signal
    # ------------------------------------------------------------------
    def _critic_grad_wrt_action(self, obs: th.Tensor, a_mu: th.Tensor) -> th.Tensor:
        """w = dQ_min/da at the mean action, DETACHED.

        Taken w.r.t. a fresh action leaf, so no critic or actor parameter receives a
        gradient from this query — it only reads the critic's slope in action space. The
        min over the twin critics matches the Q used in the standard actor term.
        """
        with th.enable_grad():
            leaf = a_mu.detach().clone().requires_grad_(True)
            q = th.cat(self.critic(obs, leaf), dim=1)          # (B, 2)
            q_min, _ = th.min(q, dim=1, keepdim=True)          # (B, 1)
            w, = th.autograd.grad(q_min.sum(), leaf)           # (B, A)
        return w.detach()

    # ------------------------------------------------------------------
    # Training — SB3 2.7.0 SAC.train() verbatim, plus the bypass in the actor loss
    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        # Bypass diagnostics (empty when bypass_off).
        bypass_grad_ratios = []          # ||grad_actor bypass|| / ||grad_actor SAC||
        # Same quantities split per action dim — each entry is an (A,) array.
        per_dim_mask, per_dim_w, per_dim_w_abs = [], [], []
        per_dim_sat, per_dim_into = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer. sample_with_blame() draws indices exactly as
            # BaseBuffer.sample() does, so both arms consume the same RNG and see the same
            # batch distribution; the baseline just ignores the blame column.
            replay_data = self.replay_buffer.sample_with_blame(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.entropy_off:
                # alpha = 0: no entropy bonus in the actor loss and no entropy term in the
                # critic target (see below). Actor loss collapses to -Q — standard SAC with
                # the entropy regularizer removed. The saved auto-entropy tuner is left in
                # place but unused, so this is a clean on/off with nothing else changed.
                ent_coef = th.zeros(1, device=self.device)
            elif self.entropy_frozen and self.log_ent_coef is not None:
                # alpha FROZEN at the loaded (warm-checkpoint) value: still applied as the
                # entropy weight, but never tuned (ent_coef_loss stays None, so the optimizer
                # step below is skipped and log_ent_coef never moves). Keeps a modest entropy
                # regularizer without letting the tuner crank alpha up to fight the bypass.
                ent_coef = th.exp(self.log_ent_coef.detach())
            elif self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient (alpha)
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critic (NOT frozen — this is co-training)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss (standard SAC)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())          # SAC part only, kept interpretable

            # --- gradient-direction bypass (masked), added to the actor loss ----------
            if not self.bypass_off:
                mu, _, _ = self.actor.get_action_dist_params(replay_data.observations)
                a_mu = th.tanh(mu)
                w = self._critic_grad_wrt_action(replay_data.observations, a_mu)
                mask = self._bypass_mask(a_mu, replay_data.blame)      # (B, A), 0/1
                bypass_term = -(mask * w * mu).sum(dim=1).mean()
                actor_params = list(self.actor.parameters())
                g_sac = th.autograd.grad(actor_loss, actor_params,
                                         retain_graph=True, allow_unused=True)
                g_byp = th.autograd.grad(bypass_term, actor_params,
                                         retain_graph=True, allow_unused=True)
                zero = th.zeros((), device=mu.device)
                sac_norm = th.sqrt(sum((g.detach().pow(2).sum() for g in g_sac if g is not None), zero))
                byp_norm = th.sqrt(sum((g.detach().pow(2).sum() for g in g_byp if g is not None), zero))
                bypass_grad_ratios.append((byp_norm / (sac_norm + 1e-8)).item())

                actor_loss = actor_loss + bypass_term
                # Per-dim, so steering and throttle can be read apart. `into_wall` is the
                # share of selected entries the bypass drives TOWARD the bound (w agrees
                # with the sign of a_mu); the rest it pulls back out of saturation, which
                # it does with the same 1/(1-a^2) amplification.
                with th.no_grad():
                    n_d = mask.sum(dim=0).clamp(min=1)                 # (A,) selected per dim
                    into = ((w * a_mu.sign()) > 0).float() * mask
                    per_dim_mask.append(mask.mean(dim=0).cpu().numpy())
                    # RAW signed mean of w over the selected entries. Reads direction
                    # directly, but can cancel: a dim with equal push on its + and -
                    # shoulders averages to ~0 while the critic is asking hard on both.
                    per_dim_w.append((w * mask).sum(dim=0).div(n_d).cpu().numpy())
                    # Magnitude, immune to that cancellation. Read the two together.
                    per_dim_w_abs.append((w.abs() * mask).sum(dim=0).div(n_d).cpu().numpy())
                    per_dim_sat.append((a_mu.abs() >= self.a_on).float().mean(dim=0).cpu().numpy())
                    per_dim_into.append(into.sum(dim=0).div(n_d).cpu().numpy())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # Bypass diagnostics (bypass arm only). bypass_grad_ratio is how hard the bypass
        # bends the actor update vs standard SAC. "Is the bypass inert?" is read off the
        # per-dim mask_frac pair below — both steer and throttle at 0 => silently a baseline.
        if bypass_grad_ratios:
            self.logger.record("train/bypass_grad_ratio", float(np.mean(bypass_grad_ratios)))
            # Per-dim split — steering and throttle read apart. mask_frac says which control
            # the bypass acts on; into_wall_frac says which way it drives it (1.0 = always
            # toward the bound, 0.0 = always back out of saturation).
            means = {
                "mask_frac": np.mean(per_dim_mask, axis=0),
                "w_mean": np.mean(per_dim_w, axis=0),
                "w_abs": np.mean(per_dim_w_abs, axis=0),
                "sat_frac": np.mean(per_dim_sat, axis=0),
                "into_wall_frac": np.mean(per_dim_into, axis=0),
            }
            for d, name in enumerate(self._dim_names()):
                for key, arr in means.items():
                    self.logger.record(f"train/{name}_{key}", float(arr[d]))
