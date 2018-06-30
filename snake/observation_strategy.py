from abc import abstractmethod

import gym


class ObservationStrategy:
    def __init__(self, game: 'snake.game'):
        self.game = game

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def observe(self):
        pass
