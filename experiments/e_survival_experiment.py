import gym

from experiments import a_basic_experiment
from snake import Game, DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(10, 10))
        game = SurvivalFitnessWrapper(game)
        return game


class SurvivalFitnessWrapper(gym.RewardWrapper):
    def __init__(self, env: Game):
        super(SurvivalFitnessWrapper, self).__init__(env)
        self.env = env

    @property
    def info_fields(self):
        return self.env.info_fields

    def reward(self, reward):
        return self.env.num_steps ** 2 * 2


if __name__ == '__main__':
    ExperimentRunner.run()
