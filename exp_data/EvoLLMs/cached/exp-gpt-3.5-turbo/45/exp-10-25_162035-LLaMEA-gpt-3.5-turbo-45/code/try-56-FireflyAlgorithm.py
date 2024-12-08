import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.2
        self.beta = 1.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

    def attractiveness(self, xi, xj):
        return np.exp(-self.beta * np.linalg.norm(xi - xj))

    def move_fireflies(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[j]) < func(self.population[i]):
                    r = np.linalg.norm(self.population[i] - self.population[j])
                    beta_i = self.beta * np.exp(-self.alpha * r**2)
                    self.population[i] += beta_i * (self.population[j] - self.population[i]) + np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            self.move_fireflies(func)

        return self.population[np.argmin([func(ind) for ind in self.population])]
    