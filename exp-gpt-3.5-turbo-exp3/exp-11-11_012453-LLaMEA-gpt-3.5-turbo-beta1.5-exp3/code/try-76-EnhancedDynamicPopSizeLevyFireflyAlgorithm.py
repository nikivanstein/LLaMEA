import numpy as np

class EnhancedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 1.0  # Dimensional adaptation factor

    def levy_flight(self):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta) * self.alpha
        return sigma2

    def levy_update(self, x):
        step = self.levy_flight()
        new_x = x + step * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, self.lb, self.ub)