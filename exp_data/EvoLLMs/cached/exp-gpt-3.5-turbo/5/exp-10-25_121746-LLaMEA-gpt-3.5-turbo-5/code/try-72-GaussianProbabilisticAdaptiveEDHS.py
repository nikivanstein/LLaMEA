import numpy as np

class GaussianProbabilisticAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on a Gaussian distribution with a probability of 0.05
            if np.random.rand() < 0.05:
                self.mutation_rate = np.random.normal(0.3, 0.1)  # Mean 0.3, Standard deviation 0.1
            super().__call__(func)
        return self.get_global_best()