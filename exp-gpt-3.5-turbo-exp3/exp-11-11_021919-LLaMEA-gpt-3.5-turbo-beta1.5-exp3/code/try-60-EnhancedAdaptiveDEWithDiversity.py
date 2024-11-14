import numpy as np

class EnhancedAdaptiveDEWithDiversity(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.3, 0.7
        self.CR_min, self.CR_max = 0.4, 0.6
        self.diversity_factor = 0.5

    def diversity_maintenance(self, pop):
        # Implement additional diversity maintenance mechanism
        ...

    def __call__(self, func):
        return super().__call__(func)