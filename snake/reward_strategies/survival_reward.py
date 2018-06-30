from ..reward_strategy import RewardStrategy


class SurvivalRewardStrategy(RewardStrategy):
    def reward(self):
        return self.game.timesteps ** 2 * 2
