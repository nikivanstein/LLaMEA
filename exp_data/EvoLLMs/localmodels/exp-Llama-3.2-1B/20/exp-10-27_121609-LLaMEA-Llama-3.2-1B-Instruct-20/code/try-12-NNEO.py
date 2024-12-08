import numpy as np

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.selection_prob = 0.2

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            new_individual = individual.copy()
            if np.random.rand() < self.selection_prob:
                idx = np.random.randint(0, self.dim)
                new_individual[idx] += np.random.uniform(-1.0, 1.0)
            return new_individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = objective(self.population[i])
                if fitness < self.fitnesses[i, self.population[i]] + 1e-6:
                    self.fitnesses[i, self.population[i]] = fitness
                    self.population[i] = mutate(self.population[i])

        return self.fitnesses

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import numpy as np
# import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.selection_prob = 0.2

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            new_individual = individual.copy()
            if np.random.rand() < self.selection_prob:
                idx = np.random.randint(0, self.dim)
                new_individual[idx] += np.random.uniform(-1.0, 1.0)
            return new_individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = objective(self.population[i])
                if fitness < self.fitnesses[i, self.population[i]] + 1e-6:
                    self.fitnesses[i, self.population[i]] = fitness
                    self.population[i] = mutate(self.population[i])

        return self.fitnesses