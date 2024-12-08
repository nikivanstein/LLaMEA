import numpy as np

class CustomRefinedEnhancedHybridCuckooDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25

    def __call__(self, func):
        self.pa = np.clip(self.pa * 1.05, 0, 1)
        return EnhancedHybridCuckooDE(self.budget, self.dim, pa=self.pa)(func)