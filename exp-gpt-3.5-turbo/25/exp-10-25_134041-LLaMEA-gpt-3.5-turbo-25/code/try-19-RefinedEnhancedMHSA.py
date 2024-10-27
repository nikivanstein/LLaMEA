import numpy as np

class RefinedEnhancedMHSA(EnhancedMHSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def harmony_search_move(x, best, bandwidth=0.05):
            r = np.random.rand(self.dim)
            prob = np.random.rand(self.dim)
            x = np.where(prob < 0.25, (1 - bandwidth) * x + bandwidth * best + bandwidth * r, np.random.uniform(-5.0, 5.0, size=(self.dim)))
            x = np.clip(x, -5.0, 5.0)
            return x

        return super().__call__(func)