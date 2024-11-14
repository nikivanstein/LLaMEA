import numpy as np

class ImprovedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 1.0  # Step size control parameter

    def update_step_size(self):
        self.beta0 = self.beta0 * np.exp(self.alpha * (np.random.rand() - 0.5))

    def levy_flight(self):
        self.update_step_size()
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
        return sigma2