# SAC tanh Jacobian bypass

Companion code for the note:

> *Unthrottling the Tanh Jacobian in SAC: A Negative Result on Bang-Bang Control and MetaDrive*

SAC squashes actions with $\tanh$, so the critic gradient that reaches the pre-tanh mean is multiplied by $1-a^2$ and vanishes near the bounds. This repo adds one extra actor term whose gradient on that mean is the *unthrottled* $\partial Q/\partial a$, then compares it to an otherwise identical SAC baseline.

The headline result is negative. On a minimum-time double integrator whose optimum is bang-bang at $\pm 1$, vanilla SAC is already near-optimal. An ungated bypass saturates the action and collapses return. A gated bypass (shoulder $|a|\in[0.9, 0.999]$) also fails, without leaving a saturated policy.

## Layout

```
toy_bangbang.py              # cart toy
check_bypass.py              # unit checks for the extra term
controllers/sb3/             # SACBypass + helpers
toy_results_gated/           # 10-seed gated toy
toy_results_ungated/         # 10-seed ungated toy
```

`SACBypass` is a thin Stable-Baselines3 SAC subclass. `bypass_off=True` drops the extra term and leaves collection, replay, critic, and entropy unchanged.

## Setup

Python 3.10+, then:

```bash
pip install "stable-baselines3" gymnasium torch numpy tqdm
```

## Sanity check

```bash
python check_bypass.py
```

No simulator. Confirms the bypass gradient on the pre-tanh mean is $-\partial Q/\partial a$ (unthrottled), that the critic is still updated, and that `--bypass-off` changes only the actor path.

## Toy

CPU, minutes to tens of minutes. Both arms share seeds; only the loss term differs.

Reference table only:

```bash
python toy_bangbang.py --calibrate
```

Smoke test (3 seeds × 12k steps):

```bash
python toy_bangbang.py --quick --out toy_results_smoke
```

Paper runs (10 seeds × 40k steps):

```bash
python toy_bangbang.py --gated --out toy_results_gated
python toy_bangbang.py --out toy_results_ungated
```

`--gated` is shorthand for `--a-on 0.9 --delta-wall 0.001`. Omitting it leaves the bypass ungated (`a_on=0`, `delta_wall=0`).

Each output directory contains `summary.txt` and `curves.csv`. Read `summary.txt` first. Four outcomes:

| Observation | Reading |
| --- | --- |
| Higher saturation **and** better return | throttle is real and it matters |
| Higher saturation, same return | throttle is real but harmless |
| Neither moves | term is inert |
| Baseline already near the bound, bypass worse | throttle was not the constraint |

Logged toy results match the last row.
