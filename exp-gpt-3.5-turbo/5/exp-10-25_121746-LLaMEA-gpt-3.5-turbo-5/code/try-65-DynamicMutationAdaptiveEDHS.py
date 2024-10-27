import numpy as np

class DynamicMutationAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            if np.random.rand() < 0.05:  # Probability-based mutation rate adjustment
                self.mutation_rate = np.random.uniform(0.1, 0.5)
            super().__call__(func)
        return self.get_global_best()