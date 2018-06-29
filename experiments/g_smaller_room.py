import snake
from experiments import a_basic_experiment
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = DistanceObservationGame(map_size=(30, 30), initial_snake_length=3)
        game = a_basic_experiment.FitnessWrapper(game)
        return snake.Agent(env=game)


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
