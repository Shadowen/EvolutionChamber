from abc import abstractmethod


class RewardStrategy:
    """Object used to determine the reward of a Snake game."""

    def __init__(self, game: 'snake.game'):
        self.game = game

    @abstractmethod
    def reward(self) -> int:
        pass
