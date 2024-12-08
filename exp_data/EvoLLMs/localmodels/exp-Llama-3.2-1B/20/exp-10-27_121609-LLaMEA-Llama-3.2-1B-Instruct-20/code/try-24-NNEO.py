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

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = np.zeros((self.population_size, self.dim, self.budget))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for i in range(self.budget):
            for j in range(self.population_size):
                x = self.population[j]
                fitness = objective(x)
                if fitness < self.fitnesses[j, x] + 1e-6:
                    self.fitnesses[j, x] = fitness
                    self.population[j] = x

        return self.fitnesses

class EvolutionaryOptimizationWithMutation(NNEO):
    def __init__(self, budget, dim, mutation_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate

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
                    self.population[i] = x

        # Apply mutation
        for i in range(self.population_size):
            x = self.population[i]
            if random.random() < self.mutation_rate:
                x += random.uniform(-1.0, 1.0)
                x = np.clip(x, bounds[x].min(), bounds[x].max())

        return self.fitnesses

class EvolutionaryOptimizationWithCrossover(NNEO):
    def __init__(self, budget, dim, crossover_rate):
        super().__init__(budget, dim)
        self.crossover_rate = crossover_rate

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x1 = self.population[i]
                x2 = self.population[i + 1]
                if random.random() < self.crossover_rate:
                    x = (1.0 - self.crossover_rate) * x1 + self.crossover_rate * x2
                    x = np.clip(x, bounds[x].min(), bounds[x].max())
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

# One-line description:
# Novel metaheuristic algorithm for black box optimization using evolutionary strategies with mutation and crossover.

# Code: 