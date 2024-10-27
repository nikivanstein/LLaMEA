import numpy as np
import random
from scipy.optimize import minimize

class AdaptiveMutationNNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_rate = 0.2
        self.mutations = 0

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if random.random() < self.mutation_rate:
                idx = random.randint(0, self.dim - 1)
                x[idx] = random.uniform(-5.0, 5.0)
            return x

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutate(x)

        return self.fitnesses

    def evaluate_fitness(self, func, problem):
        new_individual = func(problem)
        return self.__call__(new_individual)