import numpy as np

class HybridBatPSOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.velocity = np.zeros((budget, dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, dim)
        self.best_fitness = float('inf')
        self.A = 1.0
        self.r = 0.5
        self.alpha = 0.9
        self.gamma = 0.9

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                new_solution = self.population[i] + self.velocity[i]
                new_solution = np.clip(new_solution, -5.0, 5.0)
                fitness_new = func(new_solution)
                if fitness_new < func(self.population[i]):
                    self.population[i] = new_solution
                    if fitness_new < self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = fitness_new
                if np.random.rand() < self.r:
                    new_solution = self.best_solution + self.A * np.random.uniform(-1, 1, self.dim)
                    new_solution = np.clip(new_solution, -5.0, 5.0)
                    fitness_new = func(new_solution)
                    if fitness_new < self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = fitness_new
                self.velocity[i] = self.alpha * self.velocity[i] + self.gamma * (self.best_solution - self.population[i])

        return self.best_solution