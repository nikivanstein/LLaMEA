import numpy as np

class DynamicBoundaryProbabilisticAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on a time-varying strategy with a probability and dynamic boundary
            if np.random.rand() < 0.05:
                self.mutation_rate = np.random.uniform(max(0.1, self.mutation_rate - 0.1), min(0.5, self.mutation_rate + 0.1))
            super().__call__(func)
        return self.get_global_best()