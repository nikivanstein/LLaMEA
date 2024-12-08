import numpy as np

class FireflyAlgorithmDynamicAttraction:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.2
        self.beta_min = 0.2
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

    def attractiveness(self, x, y, t):
        return self.beta_min * np.exp(-self.alpha * np.linalg.norm(x - y)**2)

    def move_fireflies(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[i]) > func(self.population[j]):
                    attractiveness_ij = self.attractiveness(self.population[i], self.population[j], t)
                    self.population[i] += attractiveness_ij * (self.population[j] - self.population[i])

    def __call__(self, func):
        for t in range(self.budget):
            self.move_fireflies(func)

        return self.population[np.argmin([func(ind) for ind in self.population])]