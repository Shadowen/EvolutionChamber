import numpy as np
from gym import spaces

from snake.game import Game
from snake.observation_strategy import ObservationStrategy


class RGBObservationStrategy(ObservationStrategy):
    def __init__(self, game: Game):
        super().__init__(game)
        self._observation_space = spaces.Box(low=0, high=1, shape=self.game.map_size.tolist() + [3], dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    def observe(self):
        return self.game.render(mode='rgb_array')
