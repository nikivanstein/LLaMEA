import numpy as np

class RefinedProbabilisticAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            if np.random.rand() < 0.05:
                self.mutation_rate = np.random.uniform(0.1, 0.5)  # Adjust mutation rate with a 5% probability
            super().__call__(func)
        return self.get_global_best()