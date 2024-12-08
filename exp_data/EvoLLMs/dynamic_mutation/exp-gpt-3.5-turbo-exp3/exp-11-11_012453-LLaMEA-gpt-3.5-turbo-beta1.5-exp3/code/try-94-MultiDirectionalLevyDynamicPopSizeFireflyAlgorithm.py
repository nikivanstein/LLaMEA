import numpy as np

class MultiDirectionalLevyDynamicPopSizeFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 1.5  # Levy flight step size exponent

    def multi_directional_levy_flight(self):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim + 1), -beta))) ** (1 / self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1 / beta)
        angles = np.random.uniform(0, 2*np.pi, size=self.dim)
        step = np.multiply(sigma2, [np.cos(angle) for angle in angles]), np.multiply(sigma2, [np.sin(angle) for angle in angles])
        return step

    def levy_update(self, x):
        step = self.multi_directional_levy_flight()
        new_x = x + step
        return np.clip(new_x, self.lb, self.ub)