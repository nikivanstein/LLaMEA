import numpy as np

class RefinedEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            self.mutation_rate = self.mutation_rate * np.exp(-0.02 * _)  # Adjusted time-varying strategy
            super().__call__(func)
        return self.get_global_best()