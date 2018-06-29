import snake
from experiments import a_basic_experiment
from gym_util import MaxTimestepsWrapper
from gym_util import RecordRewardWrapper
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = DistanceObservationGame(map_size=(20, 20))
        game = SurvivalFitnessWrapper(game)
        game = MaxTimestepsWrapper(game, max_timesteps=1000)
        game = RecordRewardWrapper(game)
        return snake.Agent(env=game, hidden_nodes=[18, 18])


class SurvivalFitnessWrapper(ForwardingRewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward=0):
        return self.env.timesteps ** 2 * 2


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
