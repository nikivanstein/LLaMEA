import numpy as np

class ImprovedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 1.0  # Initial attraction coefficient

    def attractiveness(self, x_i, x_j):
        r = np.linalg.norm(x_i - x_j)
        return self.alpha * np.exp(-r)

    def levy_update(self, x_i, x_j):
        step = self.levy_flight()
        new_x = x_i + step * np.random.normal(0, 1, self.dim) + self.attractiveness(x_i, x_j) * (x_j - x_i)
        return np.clip(new_x, self.lb, self.ub)