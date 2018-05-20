from gym_util.forwarding_wrappers import ForwardingWrapper


class MaxTimestepsWrapper(ForwardingWrapper):
    def __init__(self, env, *, max_timesteps):
        super(MaxTimestepsWrapper, self).__init__(env)
        self.max_timesteps = max_timesteps

    def step(self, action):
        ob, reward, done, info = super(MaxTimestepsWrapper, self).step(action)
        if self.timesteps > self.max_timesteps:
            done = True
        return ob, reward, done, info
