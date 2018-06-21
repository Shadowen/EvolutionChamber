from typing import Tuple, Callable, List

import gym
import numpy as np

from genetic.genome import Genome


class Agent:
    def __init__(self, *,
                 env: gym.Env,
                 build_agent: Callable[
                     [gym.Space, gym.Space], Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]]):
        self.env = env

        self.get_action, self.genome = build_agent(self.env.observation_space, self.env.action_space)

    def run_iteration(self, *, render: bool = False) -> Tuple[float, List]:
        obs = self.env.reset()
        done = False
        while not done:
            action = self.get_action(self.genome, obs)
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render(mode='human')
        return reward, info
