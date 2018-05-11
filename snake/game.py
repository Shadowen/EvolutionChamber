from collections import deque
from typing import Iterable

import numpy as np
import pygame
from gym import Env, spaces

from snake.direction import Direction

pygame.init()


class Game(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 4
    }
    reward_range = (-np.inf, np.inf)

    action_space = spaces.Discrete(4)

    def __init__(self, *, map_size: Iterable[int]):
        self.initial_snake_length: int = 3
        self.map_size: np.ndarray = np.array(map_size)
        self.render_scale: int = 30

        self.observation_space = spaces.Box(low=np.array([0, 0] * 3), high=np.array(self.map_size.tolist() * 3),
                                            dtype=np.float32)

        # Snake.
        self.snake_position: np.ndarray = None
        self.snake_direction: Direction = None
        self.snake_length: int = None
        self.snake_tail: deque[np.ndarray[int]] = None

        # Food.
        self.food_position: np.ndarray[int] = None

        # Rendering.
        self.window: pygame.SurfaceType = None

    def observation(self):
        """A minimal observation. Override this as appropriate."""
        return [self.snake_position, self.snake_tail, self.food_position]

    def reset(self):
        self.snake_position = np.array(self.map_size / 2, dtype=int)
        self.snake_direction = np.array(list(Direction)[np.random.randint(0, len(Direction))].value)
        self.snake_length = self.initial_snake_length

        self.snake_tail = deque()

        self.food_position = self._get_free_position()
        return self.observation()

    def step(self, action: Direction):
        self.snake_direction = np.array(action.value)

        # Update tail.
        self.snake_tail.append(self.snake_position.copy())
        if len(self.snake_tail) > self.snake_length:
            self.snake_tail.popleft()

        # Move head.
        self.snake_position += self.snake_direction

        # Eat food.
        food_eaten = 0
        if np.all(self.snake_position == self.food_position):
            self.food_position = self._get_free_position()
            food_eaten = 1
            self.snake_length += 1

        collision = False
        # Collide with self.
        for t in self.snake_tail:
            if np.all(self.snake_position == t):
                collision = True
                break
        # Collide with walls.
        if np.any(self.snake_position < [0, 0]) or np.any(self.snake_position >= self.map_size):
            collision = True

        return self.observation(), food_eaten, collision, {}

    def _get_free_position(self):
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
                self.window = pygame.display.set_mode(self.map_size * self.render_scale, pygame.SRCALPHA)
                pygame.display.set_caption("My window")
            self.window.fill((255, 255, 255))

            s = pygame.Surface(self.map_size, flags=pygame.SRCALPHA)
            # Head.
            pygame.draw.rect(s, (0, 0, 255), [self.snake_position, [1, 1]])
            # Tail.
            for t in self.snake_tail:
                pygame.draw.rect(s, (0, 0, 0), [t, [1, 1]])

            # Food.
            pygame.draw.rect(s, (0, 255, 0), [self.food_position, [1, 1]])

            # Scale surface to window size and blit.
            s = pygame.transform.flip(s, False, True)
            s = pygame.transform.scale(s, self.map_size * self.render_scale)
            self.window.blit(s, [0, 0])

            pygame.display.flip()
        elif mode == 'rgb_array':
            world = np.zeros(self.map_size.tolist() + [3])
            for t in self.snake_tail:
                world[t[0], t[1], :] = 255
            world[self.food_position, 1] = 255
            return world

    def close(self):
        pass
