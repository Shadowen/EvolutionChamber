from experiments import a_basic_experiment
from experiments import g_smaller_room
from snake import DistanceObservationGame, Agent


class ExperimentRunner(g_smaller_room.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = DistanceObservationGame(map_size=(20, 20), initial_snake_length=3)
        game = a_basic_experiment.FitnessWrapper(game)
        return Agent(env=game, hidden_nodes=[18])


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
