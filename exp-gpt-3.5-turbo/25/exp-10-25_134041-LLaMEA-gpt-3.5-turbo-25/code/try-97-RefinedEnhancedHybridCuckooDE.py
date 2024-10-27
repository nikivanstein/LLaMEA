import numpy as np

class RefinedEnhancedHybridCuckooDE(EnhancedHybridCuckooDE):
    def __call__(self, func):
        self.pa = np.clip(self.pa * 0.25, 0, 1)
        return super().__call__(func)