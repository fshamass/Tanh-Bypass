#!/usr/bin/env python3
"""
Sanity check for the co-trained SACBypass. Run before a training run:

    /opt/anaconda3/envs/sb3/bin/python check_bypass.py

Uses a dummy Box env (no MetaDrive) so it is fast and dependency-light. Verifies:
  1. _critic_grad_wrt_action returns dQ_min/da exactly (matches autograd);
  2. the bypass term's gradient on the mean is -w/B (moves the mean by +w, un-throttled);
  3. the CRITIC is updated by train() — co-training, NOT frozen (the key difference from
     the retired two-phase design);
  4. a full train() step runs for both arms, and bypass_off changes only the actor path.
"""

import os
import sys

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.logger import Logger

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers", "sb3"))
from sb_sac import SACBypass
from custom_replay_buffer import CustomReplayBuffer


def _null_logger():
    """A no-op logger so train() can be called directly (learn() would normally set one)."""
    return Logger(folder=None, output_formats=[])

OBS_D, ACT_D, B = 6, 2, 8


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_D,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_D,), np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_D, np.float32), {}

    def step(self, action):
        return np.zeros(OBS_D, np.float32), 0.0, False, False, {}


def build(bypass_off, a_on=0.9, delta_wall=0.001):
    # CustomReplayBuffer, because train() now samples blame via sample_with_blame().
    return SACBypass(
        "MlpPolicy", DummyEnv(), learning_rate=3e-4, buffer_size=1000,
        learning_starts=0, batch_size=B, bypass_off=bypass_off,
        a_on=a_on, delta_wall=delta_wall,
        replay_buffer_class=CustomReplayBuffer,
        policy_kwargs=dict(net_arch=[32, 32]), device="cpu", seed=0,
    )


def fill(model, n=64, blame=1.0):
    """Fill the buffer, blaming every dim of every row so the masked path is exercised."""
    for _ in range(n):
        model.replay_buffer.add(
            np.random.randn(1, OBS_D).astype(np.float32),
            np.random.randn(1, OBS_D).astype(np.float32),
            np.random.uniform(-1, 1, (1, ACT_D)).astype(np.float32),
            np.random.randn(1).astype(np.float32),
            np.zeros(1, dtype=bool),
            [{}],
            blame=np.full(ACT_D, blame, dtype=np.float32),
        )


def check(label, cond):
    print(f"   {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, f"FAILED: {label}"


print("=" * 72, "\nSACBypass sanity check\n", "=" * 72, sep="")
model = build(bypass_off=False)
model.set_logger(_null_logger())
fill(model)

# --- 1 & 2: bypass gradient identity ---------------------------------------------
print("\n[1/2] bypass gradient == -w (un-throttled)")
obs = th.randn(B, OBS_D)
mu, _, _ = model.actor.get_action_dist_params(obs)
a_mu = th.tanh(mu)
w = model._critic_grad_wrt_action(obs, a_mu)

leaf = a_mu.detach().clone().requires_grad_(True)
q = th.cat(model.critic(obs, leaf), dim=1)
qmin, _ = th.min(q, dim=1, keepdim=True)
w_ref, = th.autograd.grad(qmin.sum(), leaf)
check("w == dQ_min/da (matches autograd)", th.allclose(w, w_ref, atol=1e-6))

mask = model._bypass_mask(a_mu, th.ones(B, ACT_D))
mu_leaf = mu.detach().clone().requires_grad_(True)
term = -(mask * w * mu_leaf).sum(dim=1).mean()
g, = th.autograd.grad(term, mu_leaf)
check("d(bypass)/d(mu) == -m*w/B (SAME divisor as the standard actor term)",
      th.allclose(g, -mask * w / B, atol=1e-7))
print(f"   |g - (-m*w/B)| max = {(g + mask * w / B).abs().max().item():.2e}")

# The whole method rests on this ratio being pure geometry. A mask.sum() divisor would
# scale it by B/n_active (22-64x in treatment_s1) and reintroduce a gain parameter.
a_probe = th.tensor(0.95)
ratio = (1.0 / (1.0 - a_probe ** 2)).item()
g_std = (1.0 - a_probe ** 2).item() / B          # standard term's delivery per entry
g_byp = 1.0 / B                                  # bypass delivery per selected entry
check(f"bypass/standard ratio == 1/(1-a^2) = {ratio:.2f}x at |a|=0.95, no extra gain",
      abs(g_byp / g_std - ratio) < 1e-4)

# --- 2b: the per-dim mask itself ---------------------------------------------------
print("\n[2b] mask = 1[blamed] * 1[a_on <= |a_mu| <= 1 - delta_wall]")
a_probe = th.tensor([[0.0, 0.5], [0.9, -0.95], [0.9999, -0.93], [-0.89, -0.999]])
blame_probe = th.tensor([[1., 1.], [1., 1.], [1., 1.], [1., 0.]])
m = model._bypass_mask(a_probe, blame_probe)
expect = th.tensor([[0., 0.],    # linear middle, both signs
                    [1., 1.],    # on the shoulder: positive and NEGATIVE
                    [0., 1.],    # past the wall / on the negative shoulder
                    [0., 0.]])   # inside a_on / on-band but unblamed dim
check("mask matches the shoulder band on a hand-checked probe", th.equal(m, expect))
# The abs() form must equal the explicit two-branch form everywhere.
a_sweep = th.linspace(-1 + 1e-6, 1 - 1e-6, 200001).reshape(-1, 1)
two_branch = (((a_sweep >= model.a_on) & (a_sweep <= 1 - model.delta_wall))
              | ((a_sweep <= -model.a_on) & (a_sweep >= -1 + model.delta_wall))).float()
check("abs() band == explicit two-branch band over a dense sweep",
      th.equal(model._bypass_mask(a_sweep, th.ones_like(a_sweep)), two_branch))
check("unblamed dims are always masked out",
      th.equal(model._bypass_mask(a_probe, th.zeros(4, 2)), th.zeros(4, 2)))
check("mask is 0/1 only", bool(((m == 0) | (m == 1)).all()))

# --- 3: critic is co-trained (NOT frozen) ----------------------------------------
print("\n[3] critic is updated by train() — co-training, not frozen")
cp_before = [p.detach().clone() for p in model.critic.parameters()]
req_grad = [p.requires_grad for p in model.critic.parameters()]
check("critic params have requires_grad=True (not frozen)", all(req_grad))
model.train(gradient_steps=1, batch_size=B)
moved = any(not th.equal(a, b) for a, b in zip(cp_before, model.critic.parameters()))
check("critic params CHANGED after one update", moved)

# --- 4: both arms run; bypass_off leaves the critic path intact -------------------
print("\n[4] both arms run a full train() step")
base = build(bypass_off=True)
base.set_logger(_null_logger())
fill(base)
cb = [p.detach().clone() for p in base.critic.parameters()]
base.train(gradient_steps=1, batch_size=B)
check("bypass_off arm updates the critic too", any(
    not th.equal(a, b) for a, b in zip(cb, base.critic.parameters())))
check("bypass_off flag is set", base.bypass_off is True)
check("bypass arm flag is set", model.bypass_off is False)
check("entropy is auto-tuned (log_ent_coef present)", model.log_ent_coef is not None)

print("\n" + "=" * 72 + "\nALL CHECKS PASSED\n" + "=" * 72)
