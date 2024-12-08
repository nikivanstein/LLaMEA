import numpy as np

class DynamicEnhancedHybridCuckooDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25  # Initial acceptance probability
        
    def __call__(self, func):
        self.pa = np.clip(self.pa * 1.05, 0, 1)
        return super().__call__(func)