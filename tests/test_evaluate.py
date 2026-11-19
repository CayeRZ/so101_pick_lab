"""Video frames must show the terminal state, before vectorized auto-reset."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate import TerminalFrameCapture


class OneFrameEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(1,))
    action_space = gym.spaces.Box(-1, 1, shape=(1,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.value = 1
        return np.ones(1, dtype=np.float32), 1.0, True, False, {}

    def render(self):
        return np.full((2, 2, 3), self.value, dtype=np.uint8)


def test_terminal_frame_is_preserved_after_auto_reset():
    env = DummyVecEnv([lambda: TerminalFrameCapture(OneFrameEnv())])
    env.reset()
    _, _, dones, infos = env.step(np.zeros((1, 1)))
    assert dones[0]
    assert np.all(infos[0]["terminal_frame"] == 1)
    assert np.all(env.envs[0].render() == 0)
    env.close()
