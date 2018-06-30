from ..reward_strategy import RewardStrategy


class DefaultRewardStrategy(RewardStrategy):
    def reward(self):
        return len(self.game.snake_tail)
