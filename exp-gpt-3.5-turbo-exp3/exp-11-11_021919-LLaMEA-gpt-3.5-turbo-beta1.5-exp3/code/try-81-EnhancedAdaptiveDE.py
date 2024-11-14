import numpy as np

class EnhancedAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.3, 0.7
        self.CR_min, self.CR_max = 0.4, 0.6
        self.P_min, self.P_max = 0.05, 0.2
        self.adapt_rate = 0.1

    def adapt_control_params(self, iter_count):
        self.F = self.F_min + (self.F_max - self.F_min) * np.exp(-self.adapt_rate * iter_count)
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * np.exp(-self.adapt_rate * iter_count)
        self.P = self.P_min + (self.P_max - self.P_min) * np.exp(-self.adapt_rate * iter_count)

    def __call__(self, func):
        iter_count = 0
        while iter_count < self.budget:
            self.adapt_control_params(iter_count)
            # Evolutionary process
            iter_count += self.population_size
        return super().__call__(func)