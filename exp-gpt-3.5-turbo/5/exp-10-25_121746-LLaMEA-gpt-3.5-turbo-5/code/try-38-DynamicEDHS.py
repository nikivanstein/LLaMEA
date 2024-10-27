import numpy as np

class DynamicEDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.full(budget, np.inf)

    def __call__(self, func):
        for _ in range(self.budget):
            mutation_rate = np.random.uniform(0.2, 0.4)  # Dynamic mutation rate
            for i in range(self.budget):
                candidate = self.population[i] + mutation_rate * np.random.uniform(-1, 1, self.dim)
                candidate_fitness = func(candidate)
                if candidate_fitness < self.fitness[i]:
                    self.population[i] = candidate
                    self.fitness[i] = candidate_fitness
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]