import numpy as np

class HybridDEFA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.fireflies = 20
        self.population_size = 30
        self.alpha = 0.9
        self.beta_min = 0.2
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def move_firefly(self, firefly, target):
        attraction = np.linalg.norm(firefly - target) ** 2
        step = self.beta_min * np.exp(-self.alpha * attraction) * (firefly - target)
        return firefly + step

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.fireflies):
                for j in range(self.population_size):
                    self.population[j] = self.move_firefly(self.population[j], self.population[np.random.randint(self.population_size)])
                for j in range(self.population_size):
                    if func(self.population[j]) < func(self.best_solution) or self.best_solution is None:
                        self.best_solution = np.copy(self.population[j])
        return self.best_solution