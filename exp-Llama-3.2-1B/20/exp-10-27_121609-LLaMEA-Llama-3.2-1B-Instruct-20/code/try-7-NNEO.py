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

class MutationExp:
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

        # Refine the solution
        for i in range(self.population_size):
            x = self.population[i]
            bounds_x = bounds(x)
            bounds_x[0] = -1.0
            bounds_x[1] = 1.0
            bounds_x = tuple(bounds_x)

            # Select a new line to refine the solution
            new_line = random.choice(np.random.choice(bounds_x, size=dim, replace=False))

            # Change the new line to refine the solution
            new_x = x.copy()
            new_x[new_line] += random.uniform(-0.1, 0.1)
            new_x[new_line] = np.clip(new_x[new_line], -5.0, 5.0)

            # Evaluate the new line
            new_fitness = objective(new_x)

            # If the new line is better, update the solution
            if new_fitness > self.fitnesses[i, new_line] + 1e-6:
                self.fitnesses[i, new_line] = new_fitness
                self.population[i] = new_x

        return self.fitnesses

class NNEOMut:
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

        # Refine the solution
        for i in range(self.population_size):
            x = self.population[i]
            bounds_x = bounds(x)
            bounds_x[0] = -1.0
            bounds_x[1] = 1.0
            bounds_x = tuple(bounds_x)

            # Select a new line to refine the solution
            new_line = random.choice(np.random.choice(bounds_x, size=dim, replace=False))

            # Change the new line to refine the solution
            new_x = x.copy()
            new_x[new_line] += random.uniform(-0.1, 0.1)
            new_x[new_line] = np.clip(new_x[new_line], -5.0, 5.0)

            # Evaluate the new line
            new_fitness = objective(new_x)

            # If the new line is better, update the solution
            if new_fitness > self.fitnesses[i, new_line] + 1e-6:
                self.fitnesses[i, new_line] = new_fitness
                self.population[i] = new_x

        return self.fitnesses

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 