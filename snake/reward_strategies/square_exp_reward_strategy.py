from ..reward_strategy import RewardStrategy


class SquareExpRewardStrategy(RewardStrategy):
    def reward(self):
        snake_length = self.game.snake_length + 1
        if snake_length < 10:
            return (self.game.timesteps ** 2) * (2 ** snake_length)
        else:
            return (self.game.timesteps ** 2) * (2 ** 10) * (snake_length - 9)
