from typing import List, Tuple, Any

import numpy as np

from genetic import Agent


class Distribution:
    def __init__(self, weights: List[float], items: List[Any]):
        s = sum(weights)
        self.normed_weights = [w / s for w in weights]
        self.items = items

    def sample(self, n) -> Tuple[Agent]:
        return np.random.choice(self.items, size=n, replace=True, p=self.normed_weights)
