import snake
from experiments import a_basic_experiment
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = DistanceObservationGame(map_size=(20, 20), initial_snake_length=3)
        return snake.Agent(env=game, hidden_nodes=[18, 18])


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
