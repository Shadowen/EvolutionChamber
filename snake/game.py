from collections import deque
from typing import Iterable

import numpy as np
import pygame
from gym import spaces

from snake.direction import Direction
from snake.observation_strategy import ObservationStrategy

pygame.init()


class Game:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }
    reward_range = (0, np.inf)

    action_space = spaces.Discrete(4)

    pygame_font = pygame.font.SysFont('Arial', 12)

    def __init__(self, *, map_size: Iterable[int], initial_snake_length: int = 3, observation_strategy):
        self.initial_snake_length: int = initial_snake_length
        self.map_size: np.ndarray = np.array(map_size)
        self.render_scale: int = 10

        self.observation_strategy: ObservationStrategy = observation_strategy(self)
        self.observation_space = self.observation_strategy.observation_space

        self.timesteps = None

        # Snake.
        self.snake_position: np.ndarray = None
        self.snake_direction: Direction = None
        self.snake_length: int = None
        self.snake_tail: deque[np.ndarray[int]] = None
        self.life_left: int = None

        # Food.
        self.food_position: np.ndarray[int] = None

        # Rendering.
        self.pygame_clock: pygame.time.Clock = None
        self.window: pygame.SurfaceType = None

    def reward(self):
        """The fitness function. Override this as appropriate."""
        return len(self.snake_tail)

    def reset(self):
        self.timesteps = 0

        self.snake_position = np.array(self.map_size / 2, dtype=int)
        self.snake_direction = np.array(list(Direction)[np.random.randint(0, len(Direction))].value)
        self.snake_length = self.initial_snake_length

        self.snake_tail = deque()

        self.life_left = 200

        self.food_position = self._get_free_position()
        return self.observation_strategy.observe()

    @property
    def info_fields(self):
        return ['timesteps', 'snake_length', 'life_left']

    def create_info_list(self):
        return [self.timesteps, len(self.snake_tail), self.life_left]

    def step(self, action: Direction):
        self.timesteps += 1
        self.life_left -= 1
        action_value = np.array(action.value)
        if not np.all(-action_value == self.snake_direction):
            self.snake_direction = action_value

        # Update tail.
        self.snake_tail.append(self.snake_position.copy())
        if len(self.snake_tail) > self.snake_length:
            self.snake_tail.popleft()

        # Move head.
        self.snake_position += self.snake_direction

        # Eat food.
        if np.all(self.snake_position == self.food_position):
            self.life_left += 100
            self.food_position = self._get_free_position()
            self.snake_length += 4 if self.snake_length <= 10 else 1

        done = False
        # Collide with self.
        for t in self.snake_tail:
            if np.all(self.snake_position == t):
                done = True
                break
        # Collide with walls.
        if np.any(self.snake_position < [0, 0]) or np.any(self.snake_position >= self.map_size):
            done = True

        # Run out of lifespan.
        if self.life_left <= 0:
            done = True

        return self.observation_strategy.observe(), self.reward(), done, self.create_info_list()

    def _get_free_position(self):
        if len(self.snake_tail) + 1 >= np.prod(self.map_size):
            raise NotImplementedError()  # TODO: Handle case of snake filling up entire area.

        while True:
            position = np.array([np.random.randint(self.map_size[0]), np.random.randint(self.map_size[1])])
            if np.all(position == self.snake_position):
                continue
            for t in self.snake_tail:
                if np.all(position == t):
                    break
            else:
                return position

    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            if self.window is None:
                self.pygame_clock = pygame.time.Clock()
                self.window = pygame.display.set_mode(self.map_size * self.render_scale, pygame.SRCALPHA)
                pygame.display.set_caption("Snake")
            self.window.fill((255, 255, 255))

            s = pygame.Surface(self.map_size, flags=pygame.SRCALPHA)
            # Head.
            pygame.draw.rect(s, (0, 0, 255), [self.snake_position, [1, 1]])
            # Tail.
            for t in self.snake_tail:
                pygame.draw.rect(s, (0, 0, 0), [t, [1, 1]])

            # Food.
            pygame.draw.rect(s, (0, 255, 0), [self.food_position, [1, 1]])

            # Flip snake game so it shows up properly.
            s = pygame.transform.flip(s, False, True)
            # Scale surface to window size.
            s = pygame.transform.scale(s, self.map_size * self.render_scale)

            # Add text.
            textsurface = self.pygame_font.render(f'Life left: {self.life_left}', True, (0, 0, 0))
            s.blit(textsurface, [0, 0])

            # Blit to window.
            self.window.blit(s, [0, 0])

            pygame.display.flip()
            self.pygame_clock.tick(self.metadata['video.frames_per_second'])
        elif mode == 'rgb_array':
            world = np.zeros(self.map_size.tolist() + [3])
            for t in self.snake_tail:
                world[t[0], t[1], :] = 255
            world[self.food_position, 1] = 255
            return world
