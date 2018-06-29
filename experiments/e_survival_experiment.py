import snake
from experiments import a_basic_experiment
from gym_util import MaxTimestepsWrapper
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = Game(map_size=(20, 20))
        game = MaxTimestepsWrapper(game, max_timesteps=1000)
        return snake.Agent(env=game, hidden_nodes=[18, 18])


class Game(DistanceObservationGame):
    def reward(self, reward=0):
        return self.timesteps ** 2 * 2


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
