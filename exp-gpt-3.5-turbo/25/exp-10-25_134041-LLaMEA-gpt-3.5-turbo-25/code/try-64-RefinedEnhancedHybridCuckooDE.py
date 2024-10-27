import numpy as np

class RefinedEnhancedHybridCuckooDE(EnhancedHybridCuckooDE):
    def __call__(self, func):
        self.pa = np.clip(self.pa * 1.05, 0, 1)
        return super().__call__(func)