from experiments import a_basic_experiment

import gym

from gym_util import MaxTimestepsWrapper
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(30, 30))
        game = SurvivalFitnessWrapper(game)
        game = MaxTimestepsWrapper(game, max_timesteps=10000)
        return game


class SurvivalFitnessWrapper(ForwardingRewardWrapper):
    def __init__(self, env):
        super(SurvivalFitnessWrapper, self).__init__(env)

    def reward(self, reward):
        return self.env.timesteps ** 2 * 2


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
