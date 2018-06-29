import snake
from experiments import a_basic_experiment
from gym_util import RecordRewardWrapper
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = DistanceObservationGame(map_size=(20, 20), initial_snake_length=3)
        game = BonusForReachingEndFitnessWrapper(game)
        game = RecordRewardWrapper(game)
        return snake.Agent(env=game, hidden_nodes=[18, 18])


class BonusForReachingEndFitnessWrapper(a_basic_experiment.FitnessWrapper):
    def reward(self, reward=0):
        if self.life_left == 0:
            print('BONUS')
            return super().reward() * 2
        return super().reward()


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
