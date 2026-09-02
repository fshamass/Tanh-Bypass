"""
MetaDrive -> Gymnasium wrapper with a sliding history buffer (blame queue).

Maintains a queue of the most recent transitions (newest at the front) so the training
loop can assign blame to the crash-approach window at episode end. A transition is only
known to precede a crash once the crash happens, so blame is retroactive: during the
episode the loop evicts overflow beyond history_buffer_size into the replay buffer at
blame 0; at episode end it flushes the retained window, flagged when the ending is a
crash.

Queue layout (newest at the front):
    queue[0]  = newest transition (most recent step)
    queue[-1] = oldest retained transition

Each entry: obs (before the action), action, reward, next_obs, terminated, truncated, info.

`failed` (crash OR out_of_road) is set on episode end from MetaDrive's info flags, NOT
from `terminated` (which also fires on arrive_dest success); the driver flushes the window
with a custom profile when `failed`, zeros otherwise.
"""

import gymnasium
import numpy as np
from metadrive import MetaDriveEnv
from metadrive_config import global_env_config
from bypass_common import uniform_blame_profile


class MetaDriveEnvWrapper(gymnasium.Env):
    """MetaDriveEnv with float32 obs, a seeded reset, and a sliding blame queue."""

    def __init__(self, replay_buffer, config=None):
        super().__init__()
        cfg = config or {}
        # Length of the sliding blame queue (§6.1: the last N transitions before a crash).
        self.history_buffer_size = cfg.get("history_buffer_size", 100)
        self.replay_buffer = replay_buffer
        self._md_config = {
            "use_render": cfg.get("use_render", False),
            "traffic_density": cfg.get("traffic_density", 0.5),
            "map": cfg.get("map", 3),
            "start_seed": cfg.get("start_seed", 0),
            "num_scenarios": cfg.get("num_scenarios", 1000),
            "horizon": cfg.get("horizon", 1000),
            "use_lateral_reward": cfg.get("use_lateral_reward", True),
        }
        self.md_env = MetaDriveEnv(self._md_config)
        self.observation_space = self.md_env.observation_space
        self.action_space = self.md_env.action_space
        self.a_on = global_env_config["a_on"]
        self.delta_wall = global_env_config["delta_wall"]

        self._current_obs = None
        self.queue = []
        self.failed = False

    def scenario_seed(self, seed):
        """Map an episode counter onto MetaDrive's valid scenario range (it asserts
        start_seed <= seed < start_seed + num_scenarios). Wrapping keeps every reset
        seeded and revisits the fixed scenario set once the counter laps; identity on any
        already-valid seed."""
        if seed is None:
            return None
        start = self._md_config["start_seed"]
        return start + (int(seed) - start) % self._md_config["num_scenarios"]

    def reset(self, seed=None, options=None):
        obs, info = self.md_env.reset(seed=self.scenario_seed(seed))
        obs = obs.astype(np.float32)
        self._current_obs = obs.copy()
        self.queue = []
        self.failed = False

    @staticmethod
    def _is_failure(info) -> bool:
        """A failure is a crash or going out of road (MetaDrive info flags). Excludes
        arrive_dest (success), which is also `terminated`. Matches the reference."""
        return bool(
            info.get("crash_vehicle", False)
            or info.get("crash_object", False)
            or info.get("out_of_road", False)
        )

    def step(self, action, mean_action=None):
        current_obs = self._current_obs.copy()          # obs BEFORE the action
        next_obs, reward, terminated, truncated, info = self.md_env.step(action)
        next_obs = next_obs.astype(np.float32)

        # Push newest to the front; the terminal transition is queued like any other.
        # `mean_action` is the policy's deterministic mean tanh(mu) at collection time, stored
        # so the blame flush can gate per-dim on the MEAN (stable) rather than the sampled
        # exploration action. None when the wrapper is stepped without it.
        self.queue.insert(0, {
            "obs": current_obs,
            "action": np.asarray(action, dtype=np.float32).copy(),
            "mean_action": (None if mean_action is None
                            else np.asarray(mean_action, dtype=np.float32).copy()),
            "reward": float(reward),
            "next_obs": next_obs.copy(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        })

        while len(self.queue) > self.history_buffer_size:
            entry = self.queue.pop()      # oldest transition (queue tail)
            self.add_replay_buffer_entry(entry, 
                                         blame=np.zeros(self.replay_buffer.action_dim, dtype=np.float32))

        self._current_obs = next_obs
        done = bool(terminated or truncated)
        if done:
            # crash OR out_of_road (NOT terminated, which also fires on arrive_dest).
            self.failed = self._is_failure(info)
            self.flush_episode()

        return reward, done

    def get_current_obs(self):
        return self._current_obs.copy()
    
    def flush_episode(self) -> None:
        window = list(reversed(self.queue))
        if self.failed:
            profile = uniform_blame_profile(len(window))
        else:
            profile = np.zeros(len(window), dtype=np.float32)
        for entry, b in zip(window, profile):
            # |mean action| per dim, in [0, 1); 1 where the control sat in the flat shoulder.
            ref = entry.get("mean_action")
            if ref is None:                          # fallback if stepped without a mean
                ref = entry["action"]
            a = np.abs(np.asarray(ref, dtype=np.float32))
            in_band = ((a >= self.a_on) & (a <= 1.0 - self.delta_wall)).astype(np.float32)
            self.add_replay_buffer_entry(entry, float(b) * in_band)
        self.queue = []                              # window drained; keep flush self-contained

    def add_replay_buffer_entry(self, entry, blame: np.ndarray) -> None:
        """Add one queued transition to the replay buffer with the given per-dim blame."""
        self.replay_buffer.add(
            obs=entry["obs"],
            next_obs=entry["next_obs"],
            action=entry["action"],
            reward=np.array([entry["reward"]], dtype=np.float32),
            done=np.array([entry["terminated"]]),
            infos=[entry["info"]],
            blame=blame.astype(np.float32),
        )

    def get_replay_buffer_size(self) -> int:
        return self.replay_buffer.size()

    def _rebuild_md_env(self):
        """Recreate the MetaDrive engine after an eval cycle closed it.

        MetaDrive allows only one Panda3D engine per process (engine_utils.initialize_engine
        raises otherwise), so the eval routine closes THIS env's engine, evaluates on a fresh
        held-out engine, then calls this to reopen ours. Does NOT reset — the caller issues a
        seeded reset() next, so training stays reproducible. Clears the blame queue and cached
        obs so no pre-eval state survives the rebuild.
        """
        self.md_env = MetaDriveEnv(self._md_config)
        self._current_obs = None
        self.queue = []
        self.failed = False

    def close(self):
        if self.md_env is not None:
            self.md_env.close()

def make_env(replay_buffer, traffic_density) -> MetaDriveEnvWrapper:
    """MetaDrive at the requested density, with the sliding blame queue attached."""
    cfg = {
        "use_render": False,
        "traffic_density": traffic_density,
        "map": global_env_config["map"],
        "start_seed": global_env_config["start_seed"],
        "num_scenarios": global_env_config["num_scenarios"],
        "horizon": global_env_config["horizon"],
        "use_lateral_reward": global_env_config["use_lateral_reward"],
        "history_buffer_size": global_env_config["history_buffer_size"]
    }
    return MetaDriveEnvWrapper(replay_buffer, cfg)
