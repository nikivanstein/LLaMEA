import numpy as np

class DynamicFireflyAlgorithmEEBM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5
        self.beta_min = 0.2
        self.gamma = 1.0
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def attractiveness(self, x, y):
        return np.exp(-np.sum((x - y) ** 2) / self.dim)

    def move_fireflies(self, func):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if func(self.population[j]) < func(self.population[i]):
                    beta = self.beta_min + (1 - self.beta_min) * np.random.random()
                    exploration = np.random.uniform(-5.0, 5.0, self.dim)
                    exploitation = self.alpha * (self.population[j] - self.population[i]) * self.attractiveness(self.population[i], self.population[j])
                    self.population[i] += exploration if np.random.random() < 0.35 else exploitation + self.gamma * beta * exploration

    def adapt_parameters(self, iter_count):
        self.alpha = max(0.2, self.alpha * (1 - iter_count / self.budget))
        self.gamma = min(1.0, self.gamma + iter_count / (2 * self.budget))

    def __call__(self, func):
        for iter_count in range(self.budget):
            self.adapt_parameters(iter_count)
            self.move_fireflies(func)
            for i in range(self.population_size):
                fitness = func(self.population[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = np.copy(self.population[i])
        return self.best_solution