import numpy as np

class FireflySwarmOptimization:
    def __init__(self, budget, dim, population_size=10, alpha=0.9, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, (population_size, dim))
        self.best_solution = np.copy(self.population[0])

    def __call__(self, func):
        for t in range(self.budget):
            beta = self.beta0 * np.exp(-self.gamma * t)
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness = beta / (np.linalg.norm(self.population[j] - self.population[i]) + 1e-6)
                        self.population[i] += self.alpha * attractiveness * (self.population[j] - self.population[i]) + np.random.uniform(-1, 1, self.dim)
            if func(self.population[i]) < func(self.best_solution):
                self.best_solution = np.copy(self.population[i])
        return self.best_solution