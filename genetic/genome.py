from copy import deepcopy
from typing import List

import numpy as np


class Genome:
    def __init__(self, values: List[np.ndarray]):
        self.values = values

    @classmethod
    def crossover(cls, a: 'Genome', b: 'Genome') -> 'Genome':
        """
        Crosses over two Genomes to create a third, new Genome.
        :param a: first Genome.
        :param b: second Genome.
        :param p: probability of crossover occurring.
        :return: new resultant Genome.
        """
        assert len(a.values) == len(b.values)
        assert all(np.all(x.shape == y.shape) for x, y in zip(a.values, b.values))

        c = []
        for x, y in zip(a.values, b.values):
            # Figure out what shapes we're working with.
            # We want to do this by reshaping so that we don't copy arrays.
            original_shape = x.shape
            flat_shape = np.product(x.shape)

            # Flatten x and y.
            x.shape = flat_shape
            y.shape = flat_shape
            r = np.random.randint(flat_shape)

            # Compute the crossed over matrix.
            z = np.concatenate((x[:r], y[r:]))
            # Return all the arrays to their original shapes.
            x.shape = original_shape
            y.shape = original_shape
            z.shape = original_shape
            c.append(z)

        # Assemble the full Genome from the new weights.
        return Genome(c)

    def mutate(self, p: float) -> 'Genome':
        """
        Mutates a Genome in place.
        :param p: probability of point mutation.
        :returns: self for chaining.
        """

        for i in range(len(self.values)):
            mask = np.random.choice([1, 0], size=self.values[i].shape, p=[p, 1 - p])
            mutations = np.clip(np.random.standard_normal(size=self.values[i].shape) / 5, -1, 1)
            self.values[i] = (1 - mask) * self.values[i] + mask * mutations
        return self

    def __deepcopy__(self, memodict={}):
        return Genome(deepcopy(self.values))
