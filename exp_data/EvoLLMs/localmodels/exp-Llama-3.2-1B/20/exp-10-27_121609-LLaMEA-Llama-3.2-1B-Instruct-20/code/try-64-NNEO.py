import numpy as np
import random
import math

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

        def prune(x, budget):
            if x[0] < -5.0 or x[0] > 5.0:
                return False
            if x[1] < -5.0 or x[1] > 5.0:
                return False
            for i in range(self.population_size):
                if self.fitnesses[i, x] + 1e-6 < x[0] or self.fitnesses[i, x] + 1e-6 > x[1] + 1e-6:
                    self.fitnesses[i, x] = x[0] + 1e-6
                    self.population[i] = x
            return True

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    if prune(x, self.budget):
                        self.population[i] = x

        return self.fitnesses

# Description: Evolutionary Optimization algorithm with pruning to improve search space exploration.
# Code: 
# ```python
# NNEO: (Score: -inf)
# ```
# ```python
# class NNEO:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 50
#         self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
#         self.fitnesses = np.zeros((self.population_size, self.dim))

#     def __call__(self, func):
#         def objective(x):
#             return func(x)

#         def bounds(x):
#             return (x.min() - 5.0, x.max() + 5.0)

#         def prune(x, budget):
#             if x[0] < -5.0 or x[0] > 5.0:
#                 return False
#             if x[1] < -5.0 or x[1] > 5.0:
#                 return False
#             for i in range(self.population_size):
#                 if self.fitnesses[i, x] + 1e-6 < x[0] or self.fitnesses[i, x] + 1e-6 > x[1] + 1e-6:
#                     self.fitnesses[i, x] = x[0] + 1e-6
#                     self.population[i] = x
#             return True

#         for _ in range(self.budget):
#             for i in range(self.population_size):
#                 x = self.population[i]
#                 fitness = objective(x)
#                 if fitness < self.fitnesses[i, x] + 1e-6:
#                     self.fitnesses[i, x] = fitness
#                     if prune(x, self.budget):
#                         self.population[i] = x

#         return self.fitnesses