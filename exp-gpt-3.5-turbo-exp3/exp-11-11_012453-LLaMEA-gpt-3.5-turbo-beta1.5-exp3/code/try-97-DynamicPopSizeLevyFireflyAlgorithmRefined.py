import numpy as np

class DynamicPopSizeLevyFireflyAlgorithmRefined(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def adaptive_levy_flight(self):
        beta = self.beta0 / np.sqrt(self.dim)
        sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
        sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
        
        # Adaptive mechanism to adjust step size
        if np.random.rand() < 0.5:
            sigma2 /= 2  # Reduce step size with a probability of 0.5
        
        return sigma2