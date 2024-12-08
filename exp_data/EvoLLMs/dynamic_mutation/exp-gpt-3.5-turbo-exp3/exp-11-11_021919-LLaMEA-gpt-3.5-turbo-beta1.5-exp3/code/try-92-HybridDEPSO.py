import numpy as np

class HybridDEPSO(EnhancedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.weight = 0.5

    def hybridize(self, pop, current):
        # Hybridization of DE and PSO strategies
        ...

    def __call__(self, func):
        return super().__call__(func)