import numpy as np

class ImprovedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.1  # Control parameter for population size adaptation

    def update_population_size(self, current_iter):
        self.population_size = int(self.dim * np.exp(-self.alpha * current_iter))  # Adjust population size based on iteration

    def __call__(self, func):
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        # Algorithm implementation here