import gym

from experiments import a_basic_experiment
from gym_util import MaxTimestepsWrapper
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(10, 10))
        game = MaxTimestepsWrapper(game, max_timesteps=100)
        game = SurvivalFitnessWrapper(game)
        return game


class SurvivalFitnessWrapper(ForwardingRewardWrapper):
    def __init__(self, env):
        super(SurvivalFitnessWrapper, self).__init__(env)

    def reward(self, reward):
        return self.env.timesteps ** 2 * 2


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
