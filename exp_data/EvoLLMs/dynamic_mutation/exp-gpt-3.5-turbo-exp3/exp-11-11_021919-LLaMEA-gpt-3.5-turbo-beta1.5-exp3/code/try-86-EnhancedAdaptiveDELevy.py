import numpy as np

class EnhancedAdaptiveDELevy(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.3, 0.7

    def levy_flight(self, current, best):
        beta = 1.5
        alpha = 0.01
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

    def select_parents(self, pop, current):
        # Probabilistic selection of parents strategy implementation with levy flight
        ...

    def __call__(self, func):
        return super().__call__(func)