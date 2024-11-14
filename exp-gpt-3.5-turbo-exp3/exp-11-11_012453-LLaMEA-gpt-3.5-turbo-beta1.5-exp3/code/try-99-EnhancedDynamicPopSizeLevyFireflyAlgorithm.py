import numpy as np

class EnhancedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.1  # Step size adjustment parameter

    def levy_update(self, x, global_best):
        step = self.levy_flight()
        dist_to_global_best = np.linalg.norm(x - global_best)
        adjusted_step = step * np.exp(self.alpha * dist_to_global_best)
        new_x = x + adjusted_step * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, self.lb, self.ub)