import numpy as np
import random
from scipy.optimize import minimize

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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
        return best_x, best_y

    def update_individual(self, individual, logger):
        # Refine the individual's strategy based on the number of evaluations
        if self.func_evals / self.budget < 0.15:
            # If the individual has not been improved enough, refine its strategy
            x_values = individual[:self.dim]
            y_values = individual[self.dim:]
            x_values = random.uniform(-5.0, 5.0, self.dim)
            y_values = random.uniform(-5.0, 5.0)
            logger.update_individual(x_values, y_values)
            return x_values, y_values
        else:
            # If the individual has been improved enough, return it as is
            return individual

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.

# Test the algorithm
budget = 100
dim = 10
logger = logging.getLogger(__name__)
algorithm = HyperCanny(budget, dim)
best_individual = algorithm.__call__(lambda x: x**2)
print("Best individual:", best_individual)
print("Best fitness:", best_individual[0]**2)