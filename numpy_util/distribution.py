from typing import Iterable, Tuple, Any

import numpy as np


class Distribution:
    def __init__(self, weights: Iterable[float], items: Iterable[Any]):
        s = sum(weights)
        self.normed_weights = [w / s for w in weights]
        self.items = items

    def sample(self, n) -> Tuple[Any]:
        return np.random.choice(self.items, size=n, replace=True, p=self.normed_weights)
