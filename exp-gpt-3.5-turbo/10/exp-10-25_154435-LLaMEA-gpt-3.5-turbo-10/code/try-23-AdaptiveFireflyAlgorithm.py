import numpy as np

class AdaptiveFireflyAlgorithm:
    def __init__(self, budget, dim, population_size=30, alpha=0.1, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.best_solution = self.population[np.argmin([func(x) for x in self.population])]

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(self.population[j]) < func(self.population[i]):
                        beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(self.population[j] - self.population[i]))
                        self.population[i] += beta * (self.population[j] - self.population[i]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
            self.best_solution = self.population[np.argmin([func(x) for x in self.population])]

        return self.best_solution