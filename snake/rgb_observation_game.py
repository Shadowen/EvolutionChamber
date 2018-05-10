import numpy as np
from gym import spaces

from . import Game


class RGBObservationGame(Game):
    def __init__(self, *args, **kwargs):
        super(RGBObservationGame, self).__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.map_size.tolist() + [3], dtype=np.float32)

    def observation(self):
        return self.render(mode='rgb_array')
