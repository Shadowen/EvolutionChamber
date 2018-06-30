import itertools

import numpy as np
from gym import spaces

import snake.game
from snake.observation_strategy import ObservationStrategy


class InverseDistanceObservationStrategy(ObservationStrategy):
    def __init__(self, game: snake.game.Game):
        super().__init__(game)
        self.max_distance = np.sqrt(np.sum(self.game.map_size ** 2))
        self._observation_space = spaces.Box(low=0, high=self.max_distance, shape=[8, 3],
                                             dtype=np.float32)

        self.observation_directions = [np.array(d) for d in itertools.product(*([[-1, 0, 1]] * 2)) if not d == (0, 0)]
        assert len(self.observation_directions) == 8, "observation_directions generated improperly!"

    @property
    def observation_space(self):
        return self._observation_space

    def is_occupied(self, p):
        """
        Checks the given coordinates to see if they contain an obstacle (map bounds or snake tail).
        :param p: array-like with 2 elements.
        :returns boolean representing the presence or absence of an obstacle
        """
        # Check bounds.
        if np.any(p < [0, 0]) or np.any(p >= self.game.map_size):
            return True

        # Check against snake tail.
        for t in self.game.snake_tail:
            if np.all(p == t):
                return True

    def observe(self):
        obs = []
        for direction in self.observation_directions:
            wall_distance = 0
            tail_distance = 0
            food_distance = 0
            for d in itertools.count(start=1):

                p = self.game.snake_position + direction * d

                # Check snake tail.
                if tail_distance == 0:
                    for t in self.game.snake_tail:
                        if np.all(p == t):
                            tail_distance = d

                # Check food.
                if food_distance == 0:
                    if np.all(p == self.game.food_position):
                        food_distance = d

                # Check walls.
                if wall_distance == 0:
                    if np.any(p < [0, 0]) or np.any(p >= self.game.map_size):
                        wall_distance = d
                        break

            # If tail is not present, assume max_distance.
            if tail_distance != 0:
                tail_distance = 1 / tail_distance
            # Clamp food distance to 0 or 1.
            if food_distance > 0:
                food_distance = 1
            obs.append((1 / wall_distance, tail_distance, food_distance))

        return np.array(obs)

    def reward(self, reward=0):
        snake_length = self.game.snake_length + 1
        if snake_length < 10:
            return (self.game.timesteps ** 2) * (2 ** snake_length)
        else:
            return (self.game.timesteps ** 2) * (2 ** 10) * (snake_length - 9)
