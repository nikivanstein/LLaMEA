import numpy as np

class ImprovedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.1  # Damping factor for adaptive levy flight step size

    def adaptive_levy_flight(self, diversity):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
        return sigma2 * (1 + self.alpha * diversity)

    def levy_update_with_diversity(self, x, diversity):
        step = self.adaptive_levy_flight(diversity)
        new_x = x + step * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, self.lb, self.ub)