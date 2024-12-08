import numpy as np

class CustomRefinedEnhancedHybridCuckooDE(RefinedEnhancedHybridCuckooDE):
    def __call__(self, func):
        self.pa = np.clip(self.pa * 0.25, 0, 1)  # Refine the probability adjustment
        return super().__call__(func)