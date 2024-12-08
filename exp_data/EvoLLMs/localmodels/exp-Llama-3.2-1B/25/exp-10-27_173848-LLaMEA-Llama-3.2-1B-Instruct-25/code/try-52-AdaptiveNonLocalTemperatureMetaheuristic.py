import numpy as np
import random

class AdaptiveNonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.mutations = np.random.uniform(-0.1, 0.1, size=(dim, dim))

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
                self.mutations = np.random.uniform(-0.1, 0.1, size=(dim, dim))
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation
                self.mutations = np.random.uniform(-0.1, 0.1, size=(dim, dim))

            num_evals += 1

        return self.best_func, self.mutations

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Mutation
# Code: 