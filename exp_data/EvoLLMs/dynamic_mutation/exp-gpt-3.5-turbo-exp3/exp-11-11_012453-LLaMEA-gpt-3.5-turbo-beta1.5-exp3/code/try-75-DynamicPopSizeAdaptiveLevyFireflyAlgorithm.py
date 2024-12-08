import numpy as np

class DynamicPopSizeAdaptiveLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sigma = 0.1  # Initial step size for levy flight

    def adaptive_levy_flight(self, x):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1 * self.sigma, 1/beta)
        return sigma2

    def levy_update(self, x):
        step = self.adaptive_levy_flight(x)
        new_x = x + step * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, self.lb, self.ub)