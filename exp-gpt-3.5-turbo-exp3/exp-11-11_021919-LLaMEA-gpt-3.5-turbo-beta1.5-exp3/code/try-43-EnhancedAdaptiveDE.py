import numpy as np

class EnhancedAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.2, 0.8
        self.CR_min, self.CR_max = 0.2, 0.8

    def dynamic_population_size(self, t):
        return int(10 + 40 * (1 - np.exp(-t / 800)))

    def __call__(self, func):
        return super().__call__(func)