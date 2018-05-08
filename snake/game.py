from collections import deque
from typing import List

import numpy as np
import pygame
from gym import Env

from snake.direction import Direction

pygame.init()


class Game(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-np.inf, np.inf)

    action_space = None
    observation_space = None

    def __init__(self, *, map_size: List[int]):
        self.initial_snake_length: int = 3
        self.map_size: np.ndarray = np.array(map_size)
        self.render_scale: int = 30

        # Snake.
        self.snake_position: np.ndarray = None
        self.snake_direction: Direction = None
        self.snake_length: int = None
        self.snake_tail: List[np.ndarray] = None

        # Food.
        self.food_position: np.ndarray = None

        # Rendering.
        self.window: pygame.SurfaceType = None

        self.reset()

    def reset(self):
        self.snake_position = np.array(self.map_size / 2)
        self.snake_direction = Direction.as_list()[np.random.randint(0, len(Direction))].as_np_array()
        self.snake_length = self.initial_snake_length

        self.snake_tail = deque()

        self.food_position = self._get_free_position()

    def step(self, action: Direction):
        # Don't apply a new direction if we are travelling in the opposite direction.
        # Basically don't allow the snake to instantaneously reverse direction.
        direction = action.as_np_array()
        if not np.all(direction == -self.snake_direction):
            self.snake_direction = direction

        # Update tail.
        self.snake_tail.append(self.snake_position.copy())
        if len(self.snake_tail) > self.snake_length:
            self.snake_tail.popleft()

        # Move head.
        self.snake_position += self.snake_direction

        # Eat food.
        food_eaten = False
        if np.all(self.snake_position == self.food_position):
            self.food_position = self._get_free_position()
            food_eaten = True
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

        return [self.snake_position, self.snake_tail, self.food_position], food_eaten, collision, {}

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
                self.window = pygame.display.set_mode(self.map_size * self.render_scale,
                                                      pygame.SRCALPHA)  # TODO(wheung): Window size to map size.
                pygame.display.set_caption("My window")
            self.window.fill((255, 255, 255))

            s: pygame.SurfaceType = pygame.Surface(self.map_size, flags=pygame.SRCALPHA)
            # Head.
            pygame.draw.rect(s, (0, 0, 255),
                             [self.snake_position, [1, 1]])  # TODO(wheung): Calculate grid size.
            # Tail.
            for t in self.snake_tail:
                pygame.draw.rect(s, (0, 0, 0), [t, [1, 1]])  # TODO(wheung): Calculate grid size.

            # Food.
            pygame.draw.rect(s, (0, 255, 0),
                             [self.food_position, [1, 1]])  # TODO(wheung): Calculate grid size.

            # Scale surface to window size and blit.
            s = pygame.transform.flip(s, False, True)
            s = pygame.transform.scale(s, self.map_size * self.render_scale)
            self.window.blit(s, [0, 0])

            pygame.display.flip()
        elif mode == 'rgb_array':
            world = np.zeros(self.map_size + [3])
            for t in self.snake_tail:
                world[t, :] = 255
            world[self.food_position, 1] = 255
            return world

    def close(self):
        pass


if __name__ == "__main__":
    game = Game(map_size=[10, 10])
    game.render(mode='human')
    KEY_TO_ACTION_MAP = {
        pygame.K_w: Direction.UP,
        pygame.K_a: Direction.LEFT,
        pygame.K_s: Direction.DOWN,
        pygame.K_d: Direction.RIGHT
    }


    def do_game_loop():
        update_clock = pygame.time.Clock()
        while True:
            # Render.
            game.render(mode='human')

            # Process events.
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    return
                elif event.type == pygame.KEYDOWN:
                    observation, reward, done, info = game.step(KEY_TO_ACTION_MAP[event.key])
                    if done:
                        game.reset()

            # Limit frame rate.
            update_clock.tick(30)


    do_game_loop()

pygame.quit()
