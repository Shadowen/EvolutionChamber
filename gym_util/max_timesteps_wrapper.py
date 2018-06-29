from gym import Wrapper


class MaxTimestepsWrapper(Wrapper):
    def __init__(self, env, *, max_timesteps: int):
        super().__init__(env)
        self.timesteps = 0
        self.max_timesteps = max_timesteps

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.timesteps += 1
        if self.timesteps > self.max_timesteps:
            done = True
        return ob, reward, done, info
