import numpy as np
import random

class AdaptiveBBOOPEVO:  # One-line description: Adaptive Black Box Optimization using Evolutionary Strategies
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = np.zeros((self.population_size, self.dim, self.dim))  # New addition

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population_history[i, x, 0] = fitness  # New addition
                    self.population[i] = x

        return self.fitnesses, self.population_history

    def mutate(self, func, mutation_rate):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        new_individuals = []
        for _ in range(self.population_size):
            x = self.population[random.randint(0, self.population_size - 1)]
            fitness = objective(x)
            if fitness < self.fitnesses[random.randint(0, self.population_size - 1), x] + 1e-6:
                new_individuals.append(x)

        for i in range(self.population_size):
            x = new_individuals[i]
            for j in range(self.dim):
                if random.random() < mutation_rate:
                    x[j] += random.uniform(-5.0, 5.0)
                    if x[j] < -5.0:
                        x[j] = -5.0
                    elif x[j] > 5.0:
                        x[j] = 5.0

        self.population = np.array(new_individuals)

# Description: Evolutionary strategy to optimize a black box function using adaptive mutation
# Code: 
# ```python
# AdaptiveBBOOPEVO: Evolutionary strategy to optimize a black box function using adaptive mutation
# 
# Args:
#     budget (int): Maximum number of function evaluations
#     dim (int): Dimensionality of the search space
#     mutation_rate (float): Probability of mutation in the search space
# 
# Returns:
#     None
# 
# Example:
#     adaptive_bboopevo = AdaptiveBBOOPEVO(budget=100, dim=5)
#     func = lambda x: x**2
#     adaptive_bboopevo(func, mutation_rate=0.1)