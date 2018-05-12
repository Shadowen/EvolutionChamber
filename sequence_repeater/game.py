from typing import Dict

import numpy as np
import pygame
from gym import Env, spaces


class Game(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 4
    }
    reward_range = (0, np.inf)
    info_fields = ['timesteps']

    def __init__(self, *, max_num, max_timesteps):
        # Game.
        self.max_num = max_num
        self.max_timesteps = max_timesteps
        self.correct_answer = None
        self.timesteps = None

        # Gym.
        self.observation_space = spaces.Discrete(self.max_num)
        self.action_space = spaces.Discrete(self.max_num)

        # Rendering.
        self.render_scale: int = 30
        self.window: pygame.SurfaceType = None

    def observation(self):
        ob = np.zeros(self.max_num)
        ob[self.correct_answer] = 1
        return ob

    def build_info_dict(self) -> Dict:
        return {'timesteps': self.timesteps}

    def reset(self):
        self.correct_answer = np.random.randint(4)
        self.timesteps = 1
        return self.observation()

    def step(self, action):
        done = False
        info = None
        if action == self.correct_answer:
            self.timesteps += 1
            self.correct_answer = np.random.randint(4)
        else:
            done = True

        # Prevent infinite loop.
        if self.timesteps > self.max_timesteps:
            done = True

        if done:
            info = self.build_info_dict()
        return self.observation(), self.timesteps ** 2, done, info

    def seed(self, seed=0):
        np.random.seed(seed)
