import numpy as np

class ChaosFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.alpha = 0.2
        self.beta = 2.0
        self.gamma = 0.1
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(self.population[j]) < func(self.population[i]):
                        distance = np.linalg.norm(self.population[j] - self.population[i])
                        attractiveness = 1 / (1 + self.alpha * distance**self.beta)
                        self.population[i] += attractiveness * (self.population[j] - self.population[i]) + self.gamma * np.random.uniform(-1, 1, self.dim)
        return self.population[np.argmin([func(ind) for ind in self.population])]