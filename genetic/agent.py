import itertools
from typing import Callable

import numpy as np
import tensorflow as tf
from gym import Env

from genetic.genome import Genome
from snake import Direction


class Agent:
    replica_number_counter = itertools.count()

    def __init__(self, *, env_constructor: Callable[[], Env], sess: tf.Session):
        self.replica_number: int = Agent.replica_number_counter.__next__()
        self.game = env_constructor()
        self.sess = sess

        self.observation_placeholder = tf.placeholder(shape=self.game.observation_space.shape, dtype=tf.float32)

        def build_agent(input_placeholder):
            observation_reshaped = tf.reshape(input_placeholder, shape=[1, -1])
            h1 = tf.layers.dense(inputs=observation_reshaped, units=5, activation=tf.sigmoid)
            h2 = tf.layers.dense(inputs=h1, units=8, activation=tf.sigmoid)
            output = tf.layers.dense(inputs=h2, units=self.game.action_space.n, activation=tf.nn.softmax)
            return output

        with tf.variable_scope("agent_" + str(self.replica_number)) as scope:
            self.agent_output = build_agent(self.observation_placeholder)
            self.get_variables_ops = tf.trainable_variables(scope.name)
            self.set_variables_pl = [tf.placeholder(v.dtype, shape=v.shape) for v in self.get_variables_ops]
            self.set_variables_ops = [tf.assign(v, p, validate_shape=True) for v, p in zip(
                self.get_variables_ops, self.set_variables_pl)]

    def run_iteration(self) -> float:
        self.game.reset()
        obs = self.game.render(mode='rgb_array')
        done = False
        fitness = 0
        while not done:
            action_logits = self.sess.run(self.agent_output, feed_dict={self.observation_placeholder: obs})
            action = np.random.choice(list(Direction), p=action_logits[0])
            _, reward, done, info = self.game.step(action)
            obs = self.game.render(mode='rgb_array')
            fitness += reward
        return fitness

    def set_genome(self, to: Genome):
        self.sess.run(self.set_variables_ops,
                      feed_dict={self.set_variables_pl[i]: to.values[i] for i in range(len(to.values))})

    def get_genome(self) -> Genome:
        v = self.sess.run(self.get_variables_ops)
        g = Genome(v)
        return g
