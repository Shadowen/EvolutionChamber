import gym


class ForwardingWrapper(gym.Wrapper):
    """The same as any gym.Wrapper, except missing attributes are forwarded to the wrapped environment."""

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def __getattr__(self, item):
        try:
            return self.env.__getattr__(item)
        except:
            return self.env.__getattribute__(item)


class ForwardingRewardWrapper(gym.RewardWrapper):
    """The same as any gym.RewardWrapper, except missing attributes are forwarded to the wrapped environment."""

    def reward(self, reward=0):
        return self.env.reward(reward)

    def __getattr__(self, item):
        try:
            return self.env.__getattr__(item)
        except:
            return self.env.__getattribute__(item)
