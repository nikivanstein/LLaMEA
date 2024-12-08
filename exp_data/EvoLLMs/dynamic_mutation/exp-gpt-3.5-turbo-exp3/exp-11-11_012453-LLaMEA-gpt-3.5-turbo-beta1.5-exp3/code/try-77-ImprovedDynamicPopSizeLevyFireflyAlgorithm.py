import numpy as np

class ImprovedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.1  # Population size adaptation parameter
        self.pop_size = 10  # Initial population size

    def update_population_size(self, iteration):
        self.pop_size = max(2, int(self.pop_size * (1 + self.alpha * np.exp(-iteration/self.budget))))

    def levy_update(self, x):
        step = self.levy_flight()
        new_x = x + step * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, self.lb, self.ub)