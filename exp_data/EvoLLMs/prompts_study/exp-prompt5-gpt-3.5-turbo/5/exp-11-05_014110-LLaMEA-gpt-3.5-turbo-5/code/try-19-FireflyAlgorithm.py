import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, x):
        return np.sum((x - self.population)**2, axis=1)

    def move_fireflies(self, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[j]) < func(self.population[i]):
                    self.population[i] += self.beta0 * np.exp(-self.gamma * np.linalg.norm(self.population[j] - self.population[i]))

    def __call__(self, func):
        self.population = np.random.uniform(low=-5.0, high=5.0, size=(self.budget, self.dim))
        for _ in range(self.budget):
            self.move_fireflies(func)
        return self.population[np.argmin([func(x) for x in self.population])]