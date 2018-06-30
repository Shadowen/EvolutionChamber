import snake
from experiments import a_basic_experiment
from gym_util import MaxTimestepsWrapper
from snake import Game
from snake.observation_strategies.inverse_distance_observation_strategy import InverseDistanceObservationStrategy
from snake.reward_strategies.survival_reward import SurvivalRewardStrategy


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = Game(map_size=(20, 20), create_observation_strategy=InverseDistanceObservationStrategy,
                    create_reward_strategy=SurvivalRewardStrategy)
        game = MaxTimestepsWrapper(game, max_timesteps=1000)
        return snake.Agent(env=game, hidden_nodes=[18, 18])


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
