import functools
from collections import deque
from typing import List, Iterable, Any

import numpy as np
import pygame
from gym import spaces

from snake.direction import Direction
from snake.info_emitter import InfoEmitter, PropertyEmitter
from snake.observation_strategies.default_observation_strategy import DefaultObservationStrategy
from snake.observation_strategy import ObservationStrategy
from snake.reward_strategies.default_reward_strategy import DefaultRewardStrategy
from snake.reward_strategy import RewardStrategy


class Game:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }
    reward_range = (0, np.inf)

    action_space = spaces.Discrete(4)

    def __init__(self, *, map_size: Iterable[int], initial_snake_length: int = 3,
                 create_observation_strategy=DefaultObservationStrategy, create_reward_strategy=DefaultRewardStrategy):
        # Game settings.
        self.initial_snake_length: int = initial_snake_length
        self.map_size: np.ndarray = np.array(map_size)

        self.observation_strategy: ObservationStrategy = create_observation_strategy(self)
        self.observation_space = self.observation_strategy.observation_space

        self.reward_strategy: RewardStrategy = create_reward_strategy(self)

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
        self.render_scale: int = 10
        self.pygame_clock: pygame.time.Clock = None
        self.pygame_font: pygame.font.Font = None
        self.window: pygame.SurfaceType = None

        # Info dict.
        self._info_emitters: List[InfoEmitter] = []
        self.register_info_emitter(PropertyEmitter('timesteps'))
        self.register_info_emitter(PropertyEmitter('snake_length'))
        self.register_info_emitter(PropertyEmitter('life_left'))

    def reset(self) -> Any:
        self.timesteps = 0

        self.snake_position = np.array(self.map_size / 2, dtype=int)
        self.snake_direction = np.array(list(Direction)[np.random.randint(0, len(Direction))].value)
        self.snake_length = self.initial_snake_length

        self.snake_tail = deque()

        self.life_left = 200

        self.food_position = self._get_free_position()
        return self.observation_strategy.observe()

    def register_info_emitter(self, emitter: InfoEmitter) -> None:
        self._info_emitters.append(emitter)
        self.get_info_fields.cache_clear()

    @functools.lru_cache(maxsize=1)
    def get_info_fields(self) -> List[str]:
        return [e.name for e in self._info_emitters]

    def create_info_list(self) -> List[Any]:
        return [e.emit(self) for e in self._info_emitters]

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

        return self.observation_strategy.observe(), self.reward_strategy.reward(), done, self.create_info_list()

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
                pygame.init()
                self.pygame_font = pygame.font.SysFont('Arial', 12)
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
