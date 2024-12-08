import numpy as np
import random
import copy
import math

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_rate, adaptive_line_search):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.adaptive_line_search = adaptive_line_search
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def line_search(x, y, alpha):
            return alpha * (y - x) + (1 - alpha) * math.exp(-x**2 / 2)

        def fitness(x):
            fitness = objective(x)
            if fitness < self.fitnesses[x, x] + 1e-6:
                self.fitnesses[x, x] = fitness
                return fitness
            else:
                return self.fitnesses[x, x]

        def mutate(x):
            if random.random() < self.mutation_rate:
                x[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
            return x

        def adapt_line_search(x, y):
            alpha = line_search(x, y, self.adaptive_line_search)
            return x + alpha * (y - x)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = copy.deepcopy(self.population[i])
                fitness = fitness(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    x = mutate(x)

        return self.fitnesses

# Description: 
# "Evolutionary Algorithm with Adaptive Line Search for Black Box Optimization"

# Code: 