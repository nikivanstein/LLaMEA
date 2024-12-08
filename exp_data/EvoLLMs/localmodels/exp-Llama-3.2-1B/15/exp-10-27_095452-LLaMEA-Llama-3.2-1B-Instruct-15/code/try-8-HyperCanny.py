import numpy as np

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.new_individual = None
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.random_search = False
        self.evolved_individual = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            if not self.random_search:
                x_values = np.linspace(-5.0, 5.0, 100)
                y_values = func(x_values)
                grid = dict(zip(x_values, y_values))
                best_x, best_y = None, None
                for x, y in grid.items():
                    if x < best_x or (x == best_x and y < best_y):
                        best_x, best_y = x, y
                # Random search
                if best_x is None:
                    self.random_search = True
                    self.new_individual = np.random.uniform(-5.0, 5.0, self.dim)
                    self.func_evals += 1
                    self.x_values = self.new_individual
                    self.y_values = func(self.x_values)
                    while True:
                        x_new = self.x_values + np.random.uniform(-0.1, 0.1, self.dim)
                        y_new = self.y_values + np.random.uniform(-0.1, 0.1, self.dim)
                        if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                            break
                        self.x_values = x_new
                        self.y_values = y_new
                else:
                    self.x_values = self.x_values + np.random.uniform(-0.1, 0.1, self.dim)
                    self.y_values = self.y_values + np.random.uniform(-0.1, 0.1, self.dim)
            # Evolutionary algorithm
            else:
                self.func_evals += 1
                self.x_values = np.array([x for x, y in zip(self.x_values, self.y_values) if -5.0 <= x <= 5.0])
                self.y_values = np.array([y for x, y in zip(self.x_values, self.y_values) if -5.0 <= y <= 5.0])
                for _ in range(100):
                    x_new = self.x_values + np.random.uniform(-0.1, 0.1, self.dim)
                    y_new = self.y_values + np.random.uniform(-0.1, 0.1, self.dim)
                    if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                        self.x_values = x_new
                        self.y_values = y_new
                        break
            # Check if the new solution is better
            if np.max(self.y_values) > np.max(self.y_values + 0.1):
                self.best_individual = self.x_values
                self.best_fitness = np.max(self.y_values)
                self.new_individual = np.random.uniform(-5.0, 5.0, self.dim)
                self.random_search = False
            else:
                self.best_individual = self.x_values
                self.best_fitness = np.max(self.y_values)

        return self.best_individual, self.best_fitness

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.