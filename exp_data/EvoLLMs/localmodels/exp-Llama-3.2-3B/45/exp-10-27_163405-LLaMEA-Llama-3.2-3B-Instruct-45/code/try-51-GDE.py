import numpy as np
import random

class GDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_best = np.zeros((budget, dim))
        self.f_best = np.inf
        self.x_new = np.zeros((budget, dim))
        self.f_new = np.inf
        self.p = 0.5
        self.q = 0.5

    def __call__(self, func):
        for i in range(self.budget):
            # Generate random initial population
            x = np.random.uniform(-5.0, 5.0, (self.dim, 1))

            # Evaluate initial population
            f = func(x)

            # Store best individual
            self.x_best[i] = x[0]
            self.f_best = f

            # Differential evolution
            for j in range(1, self.dim):
                for k in range(j, self.dim):
                    # Calculate difference vector
                    diff = x[k] - x[j]

                    # Calculate new individual
                    x_new = x[j] + self.q * diff

                    # Calculate new function value
                    f_new = func(x_new)

                    # Update best individual if new function value is better
                    if f_new < self.f_best:
                        self.x_best[i] = x_new
                        self.f_best = f_new

            # Update population with new best individual
            x = np.vstack((x, self.x_best))

            # Evaluate new population
            f = func(x)

            # Store best individual
            self.x_best = x[:, [np.argmin(f, axis=0)]]
            self.f_best = np.min(f, axis=0)

# Example usage:
def func(x):
    return np.sum(x**2)

gde = GDE(budget=10, dim=2)
gde(func)