import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_factor = np.full(dim, 0.5)

    def __call__(self, func):
        for _ in range(self.budget):
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]

            beta = np.random.uniform(0.5, 1.0, self.dim) * self.mutation_factor
            offspring = parent1 + beta * (parent2 - self.population)

            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

            # Update mutation factor dynamically
            self.mutation_factor = np.clip(self.mutation_factor * 1.05, 0.5, 1.0)

        return self.population[idx[0]]