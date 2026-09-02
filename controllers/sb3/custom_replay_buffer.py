"""
Replay buffer with a per-transition, PER-ACTION-DIMENSION `blame` vector.

Extends SB3's ReplayBuffer with one extra column, `blame`, holding ONE VALUE PER ACTION
DIMENSION (2 for MetaDrive: steering and throttle/brake), so a transition can be blamed
on one control and not the other. Storage of obs/action/reward/done/next_obs is the
parent's; this adds:

  - `blame`            shape (buffer_size, n_envs, action_dim)
  - add(..., blame)    blame vector at insert time, length action_dim, one entry per
                       action dim. Defaults to zeros so SB3's own collect loop (which
                       calls add() with the standard 6 args) still works.
  - sample_with_blame  standard sample plus the blame column, shape (batch, action_dim)

Deliberately minimal for now: no stratified sampling, no blamed-index list, no exit
tracking. Those are separate steps. Standard sample() is inherited unchanged, so the
trainer behaves exactly as before until blame is actually assigned and read.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class CustomReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: Optional[th.Tensor]   # carried through from the parent (n-step support)
    blame: th.Tensor                 # (batch_size, action_dim)
    batch_inds: np.ndarray


class CustomReplayBuffer(ReplayBuffer):
    """ReplayBuffer + a per-action-dim `blame` vector per transition."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        # sample_with_blame() reads next_observations directly, which the memory-optimised
        # variant does not store separately — reject it rather than branch.
        assert not optimize_memory_usage, (
            "optimize_memory_usage is not supported: this buffer stores next_observations"
        )
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # One blame value per action dimension (self.action_dim comes from the parent).
        self.blame = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        blame: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition together with its per-action-dim blame vector.

        `blame` has one entry per action dimension — shape (action_dim,) for the usual
        single-env case, or (n_envs, action_dim). It is NOT a scalar: each dim carries its
        own value. None means all-zeros, which is what SB3's own collect loop (calling this
        with the standard 6 args) gets. Blame is written at the current cursor BEFORE the
        parent advances it.
        """
        if blame is None:
            self.blame[self.pos] = 0.0
        else:
            b = np.asarray(blame, dtype=np.float32).reshape(self.n_envs, self.action_dim)
            self.blame[self.pos] = b
        super().add(obs, next_obs, action, reward, done, infos)

    @property
    def n_blamed(self) -> int:
        """Rows carrying blame > 0 on at least one action dim (diagnostic)."""
        valid = self.buffer_size if self.full else self.pos
        return int((self.blame[:valid] > 0.0).any(axis=-1).sum())

    def sample_with_blame(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> CustomReplayBufferSamples:
        """Standard uniform sample plus the blame vectors, sharing one index set."""
        upper = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper, size=batch_size)
        data = self._get_samples(batch_inds, env=env)     # parent ReplayBufferSamples
        # Single-env collection (this project's loop), so env axis 0; (batch, action_dim).
        blame = self.to_torch(self.blame[batch_inds, 0, :])
        # Reference fields by name (the parent's field set varies across SB3 versions).
        return CustomReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards,
            discounts=getattr(data, "discounts", None),
            blame=blame,
            batch_inds=batch_inds,
        )
