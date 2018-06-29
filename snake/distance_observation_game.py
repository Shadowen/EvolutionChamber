import itertools

import numpy as np
from gym import spaces

from snake import Game


class DistanceObservationGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_distance = np.sqrt(np.sum(self.map_size ** 2))
        self.observation_space = spaces.Box(low=0, high=self.max_distance, shape=[8, 3],
                                            dtype=np.float32)

        self.observation_directions = [np.array(d) for d in itertools.product(*([[-1, 0, 1]] * 2)) if not d == (0, 0)]
        assert len(self.observation_directions) == 8, "observation_directions generated improperly!"
        # Observations to return when nothing is observed in a direction.

    def is_occupied(self, p):
        """
        Checks the given coordinates to see if they contain an obstacle (map bounds or snake tail).
        :param p: array-like with 2 elements.
        :returns boolean representing the presence or absence of an obstacle
        """
        # Check bounds.
        if np.any(p < [0, 0]) or np.any(p >= self.map_size):
            return True

        # Check against snake tail.
        for t in self.snake_tail:
            if np.all(p == t):
                return True

    def observation(self):
        obs = []
        for direction in self.observation_directions:
            wall_distance = 0
            tail_distance = 0
            food_distance = 0
            for d in itertools.count(start=1):

                p = self.snake_position + direction * d

                # Check snake tail.
                if tail_distance == 0:
                    for t in self.snake_tail:
                        if np.all(p == t):
                            tail_distance = d

                # Check food.
                if food_distance == 0:
                    if np.all(p == self.food_position):
                        food_distance = d

                # Check walls.
                if wall_distance == 0:
                    if np.any(p < [0, 0]) or np.any(p >= self.map_size):
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
        snake_length = self.snake_length + 1
        if snake_length < 10:
            return (self.timesteps ** 2) * (2 ** snake_length)
        else:
            return (self.timesteps ** 2) * (2 ** 10) * (snake_length - 9)

if __name__ == '__main__':
    from snake.play_human import play

    env = DistanceObservationGame(map_size=[10, 10])
    play(env=env)
