from ..reward_strategy import RewardStrategy


class ElbowedRewardStrategy(RewardStrategy):
    def __init__(self, game: 'snake.game.Game'):
        super().__init__(game)

        self.timestep_elbow = 100
        self.length_elbow = 5

    def reward(self):
        snake_length = self.game.snake_length + 1 - 4
        r1 = self.game.timesteps ** 2 if self.game.timesteps <= self.timestep_elbow else (self.timestep_elbow ** 2) * (
                self.game.timesteps - self.timestep_elbow)
        r2 = 2 ** snake_length if snake_length <= self.length_elbow else (2 ** self.length_elbow) * (
                snake_length - self.length_elbow)
        return r1 * r2
