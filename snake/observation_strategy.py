from abc import abstractmethod
from typing import Any

import gym


class ObservationStrategy:
    """Object used to create an observation for a Snake game."""

    def __init__(self, game: 'snake.game'):
        self.game = game

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def observe(self) -> Any:
        """Create the actual observation."""
        pass
