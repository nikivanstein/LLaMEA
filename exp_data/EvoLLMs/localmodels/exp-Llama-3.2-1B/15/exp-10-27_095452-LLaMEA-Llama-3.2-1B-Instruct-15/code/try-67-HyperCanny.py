import numpy as np
import random

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.x_values = None
        self.y_values = None
        self.best_x = None
        self.best_y = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            self.x_values = np.linspace(-5.0, 5.0, 100)
            self.y_values = func(self.x_values)
            grid = dict(zip(self.x_values, self.y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            self.x_values = np.random.uniform(-5.0, 5.0, self.dim)
            self.y_values = func(self.x_values)
            self.x_values = np.array([x for x, y in zip(self.x_values, self.y_values) if -5.0 <= x <= 5.0])
            self.y_values = np.array([y for x, y in zip(self.x_values, self.y_values) if -5.0 <= y <= 5.0])
            # Evolutionary algorithm
            self.func_evals += 1
            self.x_values = self.x_values + np.random.uniform(-0.1, 0.1, self.dim)
            for _ in range(100):
                self.x_values = np.random.uniform(-5.0, 5.0, self.dim)
                self.y_values = func(self.x_values)
                self.x_values = np.array([x for x, y in zip(self.x_values, self.y_values) if -5.0 <= x <= 5.0])
                self.y_values = np.array([y for x, y in zip(self.x_values, self.y_values) if -5.0 <= y <= 5.0])
            # Check if the new solution is better
            if np.max(self.y_values) > np.max(self.y_values + 0.1):
                self.best_x, self.best_y = self.x_values, self.y_values
        return self.best_x, self.best_y

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.