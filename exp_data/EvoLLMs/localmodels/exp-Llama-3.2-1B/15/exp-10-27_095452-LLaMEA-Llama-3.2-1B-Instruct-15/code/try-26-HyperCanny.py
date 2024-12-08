import numpy as np
import random

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_x = None
        self.best_y = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-5.0, 5.0, 100)
            y_values = func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-5.0, 5.0, self.dim)
            random_y_values = func(random_x_values)
            random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -5.0 <= x <= 5.0])
            random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -5.0 <= y <= 5.0])
            # Evolutionary algorithm
            self.func_evals += 1
            x_values = random_x_values
            y_values = random_y_values
            for _ in range(100):
                x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
                if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                    x_values = x_new
                    y_values = y_new
                    break
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        # Update best solution
        self.best_x, self.best_y = best_x, best_y
        # Update best fitness
        self.best_fitness = np.max(y_values)

        # Refine the strategy
        if self.best_fitness!= self.best_fitness + 0.1:
            # 15% chance of changing the individual lines of the selected solution
            if random.random() < 0.15:
                x_values = self.best_x + np.random.uniform(-0.1, 0.1, self.dim)
                y_values = self.best_y + np.random.uniform(-0.1, 0.1, self.dim)
                self.best_x = x_values
                self.best_y = y_values

        return self.best_x, self.best_y

# One-line description:
# HyperCanny: A metaheuristic algorithm for solving black box optimization problems by combining grid search, random search, and evolutionary algorithms.