import numpy as np

class AdaptiveLineSearchHyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.alpha = 0.1  # Adaptation parameter

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
            # Adaptive line search
            if np.max(y_values) > np.max(y_values + 0.1):
                if np.min(y_values) < 0.5:
                    x_values = x_values + np.random.uniform(-0.05, 0.05, self.dim)
                    y_values = y_values + np.random.uniform(-0.05, 0.05, self.dim)
                else:
                    x_values = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                    y_values = y_values + np.random.uniform(-0.1, 0.1, self.dim)
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        return best_x, best_y

# One-line description:
# AdaptiveLineSearchHyperCanny: An evolutionary algorithm with adaptive line search for solving black box optimization problems.