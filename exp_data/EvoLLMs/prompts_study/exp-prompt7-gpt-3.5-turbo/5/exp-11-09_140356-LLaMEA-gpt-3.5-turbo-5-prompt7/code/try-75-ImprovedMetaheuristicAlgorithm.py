import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_step = np.ones(dim) * 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]
            
            beta = np.random.normal(0.5, self.mutation_step, self.dim)
            offspring = parent1 + beta * (parent2 - self.population)

            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring
            
            if np.random.rand() < 0.2:  # Adaptive mutation step update
                self.mutation_step *= 0.9

        return self.population[idx[0]]