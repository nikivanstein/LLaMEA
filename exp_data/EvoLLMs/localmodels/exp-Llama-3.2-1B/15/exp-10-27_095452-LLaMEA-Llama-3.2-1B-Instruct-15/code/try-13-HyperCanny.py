import numpy as np
import random

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func = lambda x: np.max(x)

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-5.0, 5.0, 100)
            y_values = self.func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-5.0, 5.0, self.dim)
            random_y_values = self.func(random_x_values)
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
        return best_x, best_y

# HyperCanny algorithm
def hypercan_y(x, budget, dim):
    best_x, best_y = None, None
    for _ in range(budget):
        # Grid search
        x_values = np.linspace(-5.0, 5.0, 100)
        y_values = x
        grid = dict(zip(x_values, y_values))
        for x, y in grid.items():
            if x < best_x or (x == best_x and y < best_y):
                best_x, best_y = x, y
        # Random search
        random_x_values = np.random.uniform(-5.0, 5.0, dim)
        random_y_values = x
        random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -5.0 <= x <= 5.0])
        random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -5.0 <= y <= 5.0])
        # Evolutionary algorithm
        x_values = random_x_values
        y_values = random_y_values
        for _ in range(100):
            x_new = x_values + np.random.uniform(-0.1, 0.1, dim)
            y_new = y_values + np.random.uniform(-0.1, 0.1, dim)
            if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                x_values = x_new
                y_values = y_new
                break
        # Check if the new solution is better
        if np.max(y_values) > np.max(y_values + 0.1):
            best_x, best_y = x_values, y_values
    return best_x, best_y

# Test the HyperCanny algorithm
budget = 100
dim = 2
best_solution = hypercan_y(budget, dim)
print("Best solution:", best_solution)
print("Best fitness:", np.max(best_solution))