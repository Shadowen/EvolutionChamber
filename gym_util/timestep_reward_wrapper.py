from gym import RewardWrapper


class TimestepRewardWrapper(RewardWrapper):
    def __init__(self, env, timestep_reward):
        super().__init__(env)
        self.timestep_reward = timestep_reward

    def reward(self, reward):
        return reward + self.timestep_reward