import snake
from experiments import a_basic_experiment
from snake import Game
from snake.agent import Agent
from snake.observation_strategies.inverse_distance_observation_strategy import InverseDistanceObservationStrategy
from snake.reward_strategies.square_exp_reward_strategy import SquareExpRewardStrategy


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = Game(map_size=(20, 20), initial_snake_length=3,
                    create_observation_strategy=InverseDistanceObservationStrategy,
                    create_reward_strategy=SquareExpRewardStrategy)
        return snake.agent.Agent(env=game, hidden_nodes=[18, 18])


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
