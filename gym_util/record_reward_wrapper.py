import gym

from gym_util.forwarding_wrappers import ForwardingRewardWrapper


class RecordRewardWrapper(ForwardingRewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env.info_fields = env.info_fields + ['fitness']

    def create_info_list(self):
        return super().create_info_list() + self.reward()
