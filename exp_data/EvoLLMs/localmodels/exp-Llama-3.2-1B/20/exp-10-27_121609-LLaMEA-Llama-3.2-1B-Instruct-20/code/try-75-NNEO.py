# Description: Novel Hybrid Metaheuristic for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x, p):
            return x + np.random.normal(0.0, 1.0, self.dim)

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                return np.concatenate((parent1[:int(np.random.rand() * self.dim)], parent2[int(np.random.rand() * self.dim):]))
            else:
                return np.concatenate((parent2[:int(np.random.rand() * self.dim)], parent1[int(np.random.rand() * self.dim):]))

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    new_individual = crossover(x, x)
                    if random.random() < 0.2:
                        new_individual = mutate(new_individual, random.random())
                    self.population[i] = new_individual

        return self.fitnesses

# One-line description: Novel hybrid metaheuristic combining NNEO and genetic algorithm for efficient black box optimization