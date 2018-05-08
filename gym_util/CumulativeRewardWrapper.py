from gym import RewardWrapper


class CumulativeRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.accumulated_reward = 0

    def reset(self):
        self.accumulated_reward = 0
        return self.env.reset()

    def reward(self, reward):
        self.accumulated_reward += reward
        return self.accumulated_reward
