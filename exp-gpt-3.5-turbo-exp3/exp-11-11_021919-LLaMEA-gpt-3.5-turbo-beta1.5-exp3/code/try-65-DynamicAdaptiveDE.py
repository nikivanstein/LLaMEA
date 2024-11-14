import numpy as np

class DynamicAdaptiveDE(EnhancedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.3, 0.7
        self.CR_min, self.CR_max = 0.4, 0.6

    def adapt_parameters(self, pop):
        # Dynamic adaptation of F and CR based on population diversity
        diversity = np.linalg.norm(np.std(pop, axis=0))
        self.F = self.F_min + (self.F_max - self.F_min) * (diversity / self.max_diversity)
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * (diversity / self.max_diversity)

    def __call__(self, func):
        return super().__call__(func)