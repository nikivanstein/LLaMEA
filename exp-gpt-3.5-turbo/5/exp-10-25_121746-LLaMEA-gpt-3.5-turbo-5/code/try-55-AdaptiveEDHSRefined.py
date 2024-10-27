import numpy as np

class AdaptiveEDHSRefined(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on a dynamic adaptation using the function landscape
            self.mutation_rate *= np.exp(-(func(self.get_global_best()) - func(self.get_best())) / (np.linalg.norm(self.get_global_best() - self.get_best()) + 1e-8))
            super().__call__(func)
        return self.get_global_best()