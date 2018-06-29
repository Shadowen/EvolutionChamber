import snake
from experiments import a_basic_experiment
from snake import Direction
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = Game(map_size=(20, 20), initial_snake_length=3)
        return snake.Agent(env=game, hidden_nodes=[18, 18])


class Game(DistanceObservationGame):
    timestep_elbow = 100
    length_elbow = 5

    def step(self, action: Direction):
        r = super().step(action)
        self.snake_tail = []
        return r

    def reward(self, reward=0):
        snake_length = self.snake_length + 1 - 4
        r1 = self.timesteps ** 2 if self.timesteps <= self.timestep_elbow else (self.timestep_elbow ** 2) * (
                self.timesteps - self.timestep_elbow)
        r2 = 2 ** snake_length if snake_length <= self.length_elbow else (2 ** self.length_elbow) * (
                snake_length - self.length_elbow)
        return r1 * r2


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
