from gym_util.forwarding_wrappers import ForwardingWrapper


class MaxTimestepsWrapper(ForwardingWrapper):
    def __init__(self, env, *, max_timesteps):
        super(MaxTimestepsWrapper, self).__init__(env)
        self.info_fields = env.info_fields + ['reached_max_timesteps']
        self.max_timesteps = max_timesteps

    def step(self, action):
        ob, reward, done, info = super(MaxTimestepsWrapper, self).step(action)
        if self.timesteps > self.max_timesteps:
            info.append(True)
            done = True
        else:
            info.append(False)
        return ob, reward, done, info
