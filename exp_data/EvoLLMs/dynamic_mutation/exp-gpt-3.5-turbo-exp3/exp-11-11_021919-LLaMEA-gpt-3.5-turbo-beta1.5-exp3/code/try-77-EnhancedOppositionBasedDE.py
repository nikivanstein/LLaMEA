import numpy as np

class EnhancedOppositionBasedDE(EnhancedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def opposition_based_init(self, pop):
        return np.concatenate((pop, -pop))

    def __call__(self, func):
        return super().__call__(func)