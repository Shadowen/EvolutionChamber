import gym
import numpy as np

import snake.observation_strategy


class DefaultObservationStrategy(snake.observation_strategy.ObservationStrategy):
    def __init__(self, game: snake.game.Game):
        super().__init__(game)
        self._observation_space = gym.spaces.Box(low=np.array([0, 0] * 3),
                                                 high=np.array(self.game.map_size.tolist() * 3),
                                                 dtype=np.float32)

    def observe(self):
        return [self.game.snake_position, self.game.snake_tail, self.game.food_position]

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space
