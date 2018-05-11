from collections import deque

import tensorflow as tf

from genetic import Runner
from gym_util import TimestepRewardWrapper
from snake import DistanceObservationGame


class ExperimentRunner(Runner):
    @staticmethod
    def game_constructor():
        game = DistanceObservationGame(map_size=(30, 30))
        game = TimestepRewardWrapper(game, 0.1)
        return game

    @staticmethod
    def build_agent(input_placeholder, action_space):
        observation_reshaped = tf.reshape(input_placeholder, shape=[1, -1])
        h1 = tf.layers.dense(inputs=observation_reshaped, units=18, activation=tf.sigmoid)
        output = tf.layers.dense(inputs=h1, units=action_space.n, activation=tf.nn.softmax)
        return output

    @staticmethod
    def run():
        with ExperimentRunner(num_agents=20) as r:
            steps = 10000
            f_historical = deque(maxlen=100)
            for s in range(steps):
                f = r.single_iteration()
                f_historical.append(sum(f) / len(f))
                print(f"Generation {s} \t Fitness: {f_historical[-1]} (avg. {sum(f_historical) / len(f_historical)})")


if __name__ == '__main__':
    ExperimentRunner.run()
