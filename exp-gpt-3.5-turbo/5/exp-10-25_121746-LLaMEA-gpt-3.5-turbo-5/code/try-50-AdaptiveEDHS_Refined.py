import numpy as np

class AdaptiveEDHS_Refined(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.1, 0.5)  # Random mutation rate initialization

    def __call__(self, func):
        for _ in range(self.budget):
            self.mutation_rate = self.mutation_rate * np.exp(-0.01 * _)  # Update mutation rate with exponential decay
            super().__call__(func)
        return self.get_global_best()