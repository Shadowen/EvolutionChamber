import gym
from gym import RewardWrapper


class RecordRewardWrapper(RewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env.info_fields = env.info_fields + ['fitness']

    def create_info_list(self):
        return super().create_info_list() + self.reward()
