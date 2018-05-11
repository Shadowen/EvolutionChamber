import numpy as np
import pygame
from gym import Env, spaces

pygame.init()


class Game(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 4
    }
    reward_range = (0, np.inf)

    def __init__(self, max_num):
        # Game.
        self.max_num = max_num
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

    def reset(self):
        self.correct_answer = np.random.randint(4)
        self.timesteps = 1
        return self.observation()

    def step(self, action):
        done = False
        if action == self.correct_answer:
            self.timesteps += 1
            self.correct_answer = np.random.randint(4)
        else:
            done = True

        # Prevent infinite loop.
        if self.timesteps > 100:
            done = True

        return self.observation(), self.timesteps ** 2, done, {}

    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            if self.window is None:
                self.window = pygame.display.set_mode([self.max_num] * 2 * self.render_scale, pygame.SRCALPHA)
                pygame.display.set_caption("My window")
            self.window.fill((255, 255, 255))

            s = pygame.Surface([self.max_num] * 2, flags=pygame.SRCALPHA)
            # TODO: Draw stuff on  s.
            # Scale surface to window size and blit.
            s = pygame.transform.flip(s, False, True)
            s = pygame.transform.scale(s, self.map_size * self.render_scale)
            self.window.blit(s, [0, 0])

            pygame.display.flip()
        elif mode == 'rgb_array':
            raise NotImplementedError()
