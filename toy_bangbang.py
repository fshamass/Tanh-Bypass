#!/usr/bin/env python3
"""
toy_bangbang.py — the decisive test for the gradient bypass.

WHY THIS EXISTS
---------------
In MetaDrive the bypass ties or loses, but every comparison there is confounded: the warm
checkpoint decays under continued training, evaluation resolves ~4% rates from 50 episodes,
and the mask fires on ~5 of 512 rows. None of that says whether the MECHANISM works.

This file removes every confound and asks one question:

    On a task whose optimal policy is PROVABLY pinned at the action bound
    at every step, does removing the tanh gradient throttle help?

If no, the mechanism does not work and MetaDrive cannot rescue it. If yes, this is the
existence proof and the expensive runs become confirmation rather than exploration.

THE TASK — minimum-time double integrator
-----------------------------------------
    state  = (position x, velocity v)
    action = acceleration a in [-1, 1]
    reward = -1 per step, until |x| < 0.05 and |v| < 0.05

Textbook optimal control: bang-bang with a single switching curve, |a| = 1 at EVERY step.
Interior actions are never optimal, anywhere in the state space. That is the strongest
possible demand for near-bound actions, which is exactly what the bypass claims to restore.

Calibration (reference controllers, saturated at various caps — reproduce with --calibrate):

    |a| capped at 1.0   ->  -33.5 return        (optimal)
    |a| capped at 0.9   ->  -33.9
    |a| capped at 0.7   ->  -38.5
    |a| capped at 0.5   ->  -47.1
    |a| capped at 0.3   ->  -64.5

So return maps back onto an EFFECTIVE ACTION CAP. If the baseline scores -47 it is behaving
like a controller that can only reach |a| = 0.5, and the throttle is costing 40%.

WHAT IS CONTROLLED
------------------
- Both arms use the SAME `SACBypass` class the MetaDrive runs use — this exercises the
  production code path, not a reimplementation.
- The bypass is UNGATED (a_on=0, delta_wall=0, blame=1 everywhere): the v1-faithful form,
  not the blame+band-gated variant. No knobs.
- Arms are PAIRED per seed: same seed -> same init, same collection RNG.
- Seeds are genuinely INDEPENDENT here (unlike the autocorrelated MetaDrive eval series),
  so the paired t-statistic at the end is honest rather than inflated.

RUN
---
    python toy_bangbang.py              # 10 seeds x 40k steps
    python toy_bangbang.py --quick      # 3 seeds x 12k, smoke test
    python toy_bangbang.py --calibrate  # reference table only
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers", "sb3"))

from sb_sac import SACBypass                         # noqa: E402  the production class
from custom_replay_buffer import CustomReplayBuffer  # noqa: E402

DT = 0.05
MAX_STEPS = 200
TOL = 0.10          # goal box: |x| < TOL and |v| < TOL
THRESHOLD = -35.0   # "has learned the task" bar, for steps-to-threshold
RESULT_DIR = Path("toy_results")
SHAPING = 2.0       # potential-based shaping weight (0 disables); see the env docstring
GAMMA = 0.99


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MinTimeDoubleIntegrator(gym.Env):
    """Drive (x, v) to the origin in minimum time. Optimal control is bang-bang.

        v <- v + a*dt
        x <- x + v*dt
        reward = -1 every step; episode ends when |x| < TOL and |v| < TOL.

    Two-sided on purpose: x0 may be either sign, so the policy must reach BOTH action
    bounds. That matches the MetaDrive mask, which folds the two tanh shoulders together.

    POTENTIAL-BASED SHAPING. Pure minimum-time reward is -1 everywhere, so there is no
    learning signal until the agent stumbles into the goal box — a sparse-reward
    exploration problem that swamps the effect we are trying to measure. We add

        F(s, s') = GAMMA * Phi(s') - Phi(s),    Phi(s) = -SHAPING * (|x| + |v|)

    which by Ng, Harada & Russell (1999) leaves the OPTIMAL POLICY EXACTLY UNCHANGED for
    this discount factor. It makes the task learnable without touching what "optimal"
    means, so bang-bang is still the target and the bound is still where the answer lives.
    """

    metadata = {"render_modes": []}

    def __init__(self, tol=TOL, max_steps=MAX_STEPS, shaping=SHAPING):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.tol, self.max_steps, self.shaping = tol, max_steps, shaping
        self.x, self.v, self.t = 1.0, 0.0, 0

    def _obs(self):
        return np.array([self.x, self.v], dtype=np.float32)

    def _phi(self, x, v):
        return -self.shaping * (abs(x) + abs(v))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.uniform(-1.0, 1.0))
        self.v = float(self.np_random.uniform(-0.5, 0.5))
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        phi_before = self._phi(self.x, self.v)
        self.v += a * DT
        self.x += self.v * DT
        self.t += 1
        reached = abs(self.x) < self.tol and abs(self.v) < self.tol
        reward = -1.0 + (GAMMA * self._phi(self.x, self.v) - phi_before)
        return (self._obs(), reward, reached, self.t >= self.max_steps,
                {"action": a, "reached": reached})


# ---------------------------------------------------------------------------
# Replay buffer: blame = 1 everywhere  ->  the UNGATED bypass
# ---------------------------------------------------------------------------

class UngatedBlameBuffer(CustomReplayBuffer):
    """CustomReplayBuffer whose blame defaults to 1 instead of 0.

    SAC Bypass's mask is `blamed * band * dim_selector`. With blame = 1 everywhere and
    a_on = 0 / delta_wall = 0 the band is also 1 everywhere, so the mask is all-ones and
    the bypass runs UNGATED. Nothing else changes, so SB3's stock collection loop — which
    calls add() with the standard six arguments — works untouched.
    """

    def add(self, obs, next_obs, action, reward, done, infos, blame=None):
        if blame is None:
            blame = np.ones((self.n_envs, self.action_dim), dtype=np.float32)
        super().add(obs, next_obs, action, reward, done, infos, blame=blame)


# ---------------------------------------------------------------------------
# Reference controllers — the calibration scale
# ---------------------------------------------------------------------------

def reference_return(cap, grid):
    """Saturated bang-bang controller with |a| <= cap. Maps a cap onto a return."""
    steps = []
    for x0, v0 in grid:
        x, v, t = x0, v0, 0
        while t < MAX_STEPS and not (abs(x) < TOL and abs(v) < TOL):
            s = x + 0.5 * v * abs(v) / cap                    # switching function
            a = -cap * np.sign(s) if abs(s) > 1e-9 else -cap * np.sign(v)
            v += float(np.clip(a, -cap, cap)) * DT
            x += v * DT
            t += 1
        steps.append(t)
    return -float(np.mean(steps))


def eval_grid(k=9):
    """Fixed held-out starting states — identical for every arm and seed, so eval is paired."""
    return [(float(x), float(v))
            for x in np.linspace(-1.0, 1.0, k)
            for v in np.linspace(-0.5, 0.5, 5)
            if abs(x) > 1e-9 or abs(v) > 1e-9]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, grid):
    """Deterministic rollouts from the fixed grid, plus the mechanism diagnostics.

    Return is reported as -steps (the TRUE minimum-time objective), not the shaped reward,
    so it stays comparable to the reference controller table.

    The mechanism numbers are the point. The optimal policy has sat_frac == 1.0 and
    mean_abs_action == 1.0. Anything less IS the tanh throttle, measured directly.
    """
    env = MinTimeDoubleIntegrator()
    steps, reached, acts = [], [], []
    for x0, v0 in grid:
        env.reset(seed=0)
        env.x, env.v, env.t = x0, v0, 0
        obs, done, ok = env._obs(), False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            acts.append(info["action"])
            ok = info["reached"]
            done = term or trunc
        steps.append(env.t)
        reached.append(ok)
    a = np.abs(np.asarray(acts))
    return {
        "return": -float(np.mean(steps)),
        "reach_rate": float(np.mean(reached)),
        # --- mechanism: how close to the bound does the policy actually get? ---
        "mean_abs_action": float(a.mean()),
        "sat_frac": float((a >= 0.9).mean()),
        "p95_abs_action": float(np.percentile(a, 95)),
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

class EvalCurveCallback(BaseCallback):
    """Evaluate on the held-out grid every `every` env steps and record the curve.

    This is the measurement that matters. Evaluating only at the end reports the CONVERGED
    policy, and the tanh throttle scales the actor's gradient without moving the point where
    that gradient is zero — so the converged policy is the one quantity the mechanism should
    NOT affect. The difference, if there is one, lives in how fast each arm gets there.
    """

    def __init__(self, grid, every):
        super().__init__()
        self.grid, self.every, self.curve = grid, every, []

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            m = evaluate(self.model, self.grid)
            m["step"] = int(self.num_timesteps)
            self.curve.append(m)
            # Mirror into the SB3 logger so eval/* lands in TensorBoard alongside the
            # train/* scalars SACBypass.train() already records (bypass_grad_ratio,
            # dim0_mask_frac, dim0_sat_frac, ...).
            for k, v in m.items():
                if k != "step":
                    self.logger.record(f"eval/{k}", v)
            self.logger.dump(step=self.num_timesteps)
        return True


def summarise_curve(curve, budget):
    """Turn a learning curve into the two sample-efficiency numbers we compare.

    steps_to_threshold — env steps until the arm FIRST scores at or above THRESHOLD.
                         Censored at `budget` if it never gets there, so a run that never
                         learns is scored as "needed at least the whole budget", not dropped.
    auc                — mean score across the whole curve. Rewards getting good EARLY, not
                         just ending well; insensitive to where exactly the threshold is set.
    """
    hit = next((c["step"] for c in curve if c["return"] >= THRESHOLD), None)
    return {
        "steps_to_threshold": float(hit if hit is not None else budget),
        "reached_threshold": float(hit is not None),
        "auc": float(np.mean([c["return"] for c in curve])) if curve else float("nan"),
    }


def run_arm(bypass_off, seed, steps, grid, eval_every, a_on=0.0, delta_wall=0.0,
            tb_dir=None):
    model = SACBypass(
        "MlpPolicy", MinTimeDoubleIntegrator(),
        bypass_off=bypass_off,
        a_on=a_on, delta_wall=delta_wall, bypass_dims=None,
        # a_on=0, delta_wall=0  -> UNGATED: the term fires in EVERY state, including the
        #   linear region where plain SAC already delivers the gradient. No knobs, but it
        #   inflates the pre-tanh mean until tanh saturates and stays there.
        # a_on=0.9, delta_wall=0.001 -> GATED, the MetaDrive setting: the term fires only on
        #   the flat shoulder of tanh and stops at the bound, so it cannot pin the policy.
        learning_rate=3e-4, buffer_size=100_000, batch_size=256,
        learning_starts=1_000, train_freq=1, gradient_steps=1,
        replay_buffer_class=UngatedBlameBuffer,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu", seed=seed, verbose=0,
        tensorboard_log=tb_dir,
    )
    cb = EvalCurveCallback(grid, eval_every)
    arm = "baseline" if bypass_off else "bypass"
    model.learn(total_timesteps=steps, log_interval=10_000, callback=cb,
                tb_log_name=f"{arm}_s{seed}")
    final = evaluate(model, grid)
    final.update(summarise_curve(cb.curve, steps))
    final["curve"] = cb.curve
    return final


def write_curves(base, byp, path):
    """Every evaluation point from every run, so the curves can be plotted independently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["step", "return", "reach_rate", "mean_abs_action", "sat_frac", "p95_abs_action"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm"] + cols)
        for arm, runs in (("baseline", base), ("bypass", byp)):
            for seed, r in enumerate(runs):
                for pt in r["curve"]:
                    w.writerow([seed, arm] + [pt[c] for c in cols])


def paired_report(base, byp, keys, better):
    """Paired difference across INDEPENDENT seeds — this t is legitimate."""
    out = [f"\n{'metric':<18} {'baseline':>10} {'bypass':>10} {'diff':>10} {'t':>7}   verdict",
           "-" * 72]
    for k in keys:
        b = np.array([r[k] for r in base])
        y = np.array([r[k] for r in byp])
        d = y - b
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        t = d.mean() / se if se and se > 0 else np.nan
        if not np.isfinite(t) or abs(t) <= 2.0:
            verdict = "no effect"
        else:
            good = (d.mean() > 0) == (better[k] == "up")
            verdict = "BYPASS WINS" if good else "BYPASS LOSES"
        out.append(f"{k:<18} {b.mean():>10.3f} {y.mean():>10.3f} {d.mean():>+10.3f} {t:>+7.2f}   {verdict}")
    for line in out:
        print(line)
    return out


def main():
    p = argparse.ArgumentParser(description="Sample-efficiency test for the gradient bypass")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--eval-every", type=int, default=2_500,
                   help="Env steps between held-out evaluations during training.")
    p.add_argument("--a-on", type=float, default=0.0,
                   help="Mask band floor. 0.0 = ungated (fires everywhere). "
                        "0.9 = the MetaDrive gated setting (fires only on the tanh shoulder).")
    p.add_argument("--delta-wall", type=float, default=0.0,
                   help="Mask band ceiling is 1 - delta_wall. 0.0 = no ceiling. "
                        "0.001 = the MetaDrive setting (stop pushing once at the bound).")
    p.add_argument("--gated", action="store_true",
                   help="Shorthand for --a-on 0.9 --delta-wall 0.001 (the MetaDrive config).")
    p.add_argument("--quick", action="store_true", help="3 seeds x 12k steps — smoke test")
    p.add_argument("--calibrate", action="store_true", help="print the reference table and exit")
    p.add_argument("--out", type=str, default=str(RESULT_DIR),
                   help="Directory for curves.csv and summary.txt")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.steps, args.eval_every = 3, 12_000, 1_500
    if args.gated:
        args.a_on, args.delta_wall = 0.9, 0.001

    grid = eval_grid()
    lines = []

    def say(msg=""):
        print(msg)
        lines.append(msg)

    say("Reference controllers (saturated bang-bang) on the same eval grid:")
    caps = [1.0, 0.9, 0.7, 0.5, 0.3]
    refs = [(c, reference_return(c, grid)) for c in caps]
    for c, r in refs:
        say(f"    |a| <= {c:.1f}   return = {r:7.1f}")
    if args.calibrate:
        return

    def effective_cap(ret):
        xs = [r for _, r in refs][::-1]
        ys = [c for c, _ in refs][::-1]
        return float(np.interp(ret, xs, ys))

    say(f"\ntoy_bangbang: {args.seeds} paired seeds x {args.steps} steps, "
        f"eval every {args.eval_every}, {len(grid)} held-out starts")
    gate = ("UNGATED (fires everywhere)" if args.a_on == 0.0 and args.delta_wall == 0.0
            else f"GATED  a_on={args.a_on} delta_wall={args.delta_wall}")
    say(f"bypass mask: {gate}")
    say(f"threshold for steps-to-threshold: return >= {THRESHOLD}\n")

    base, byp = [], []
    for s_ in range(args.seeds):
        tb = str(Path(args.out) / "tb")
        b = run_arm(True,  s_, args.steps, grid, args.eval_every, args.a_on, args.delta_wall, tb)
        y = run_arm(False, s_, args.steps, grid, args.eval_every, args.a_on, args.delta_wall, tb)
        base.append(b)
        byp.append(y)
        say(f"  seed {s_:>2}   base: ret={b['return']:7.1f} steps2thr={b['steps_to_threshold']:>6.0f} "
            f"auc={b['auc']:7.1f}    bypass: ret={y['return']:7.1f} "
            f"steps2thr={y['steps_to_threshold']:>6.0f} auc={y['auc']:7.1f}")

    better = {"steps_to_threshold": "down", "auc": "up", "reached_threshold": "up",
              "return": "up", "reach_rate": "up", "mean_abs_action": "up",
              "sat_frac": "up", "p95_abs_action": "up"}

    say("\n=== SAMPLE EFFICIENCY (the live hypothesis: does the bypass get there FASTER?) ===")
    lines += paired_report(base, byp, ["steps_to_threshold", "auc", "reached_threshold"], better)
    say("\n=== FINAL PERFORMANCE (theory says this should NOT move) ===")
    lines += paired_report(base, byp, ["return", "reach_rate"], better)
    say("\n=== MECHANISM (does the bypass reach the bound?) ===")
    lines += paired_report(base, byp, ["mean_abs_action", "sat_frac", "p95_abs_action"], better)

    rb = float(np.mean([r["return"] for r in base]))
    ry = float(np.mean([r["return"] for r in byp]))
    say(f"\nEffective action cap:  baseline ~{effective_cap(rb):.2f}   "
        f"bypass ~{effective_cap(ry):.2f}   (optimal = 1.00)")
    say("\nHow to read this:\n"
        "  steps_to_threshold DOWN / auc UP  -> the bypass speeds learning up. The live claim.\n"
        "  sample efficiency flat            -> the mechanism does nothing; stop here.\n"
        "  final return moves                -> unexpected; the throttle is biasing the endpoint\n"
        "                                       after all, and the theory in the notes is wrong.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_curves(base, byp, out / "curves.csv")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out/'curves.csv'} and {out/'summary.txt'}")
    print(f"TensorBoard:  tensorboard --logdir {out/'tb'}")


if __name__ == "__main__":
    main()
