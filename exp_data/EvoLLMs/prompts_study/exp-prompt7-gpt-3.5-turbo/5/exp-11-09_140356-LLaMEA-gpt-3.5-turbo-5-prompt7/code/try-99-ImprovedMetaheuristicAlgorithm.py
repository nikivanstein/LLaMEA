import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_range = 0.2 * np.abs(self.population).max(axis=0)

    def __call__(self, func):
        for _ in range(self.budget):
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]
            
            beta = np.random.uniform(0.5, 1.0, self.dim)
            mutation_scale = np.random.uniform(0.1, 0.5, self.dim) * self.mutation_range
            offspring = parent1 + beta * (parent2 - self.population) * mutation_scale

            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

        return self.population[idx[0]]