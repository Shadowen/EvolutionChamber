import gym

from experiments import a_basic_experiment
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(10, 10))
        game = MaxTimestepsWrapper(game)
        game = SurvivalFitnessWrapper(game)
        return game


class MaxTimestepsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MaxTimestepsWrapper, self).__init__(env)

    @property
    def info_fields(self):
        return self.env.info_fields

    def step(self, action):
        ob, reward, done, info = super(MaxTimestepsWrapper, self).step(action)
        if self.unwrapped.timesteps > self.max_timesteps:
            done = True
        return ob, reward, done, info


class SurvivalFitnessWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(SurvivalFitnessWrapper, self).__init__(env)

    @property
    def info_fields(self):
        return self.env.info_fields

    def reward(self, reward):
        return self.env.num_steps ** 2 * 2


if __name__ == '__main__':
    ExperimentRunner.run()
