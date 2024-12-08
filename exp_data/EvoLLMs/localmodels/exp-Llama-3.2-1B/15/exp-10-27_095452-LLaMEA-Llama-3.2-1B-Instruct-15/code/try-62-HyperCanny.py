import numpy as np
import random

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.boundaries = 5.0
        self.population_size = 100
        self.population = np.random.uniform(-self.boundaries, self.boundaries, (self.population_size, self.dim))
        self.population = np.array([x + np.random.uniform(-0.1, 0.1, self.dim) for x in self.population])

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-self.boundaries, self.boundaries, 100)
            y_values = func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-self.boundaries, self.boundaries, self.dim)
            random_y_values = func(random_x_values)
            random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -self.boundaries <= x <= self.boundaries])
            random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -self.boundaries <= y <= self.boundaries])
            # Evolutionary algorithm
            self.func_evals += 1
            x_values = random_x_values
            y_values = random_y_values
            for _ in range(100):
                x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
                if -self.boundaries <= x_new <= self.boundaries and -self.boundaries <= y_new <= self.boundaries:
                    x_values = x_new
                    y_values = y_new
                    break
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        return best_x, best_y

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.