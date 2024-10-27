import numpy as np
import random
from scipy.optimize import minimize

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.mini = False
        self.best_fitness = np.inf

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
                self.best_fitness = func(new_func)
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation
                self.best_fitness = func(new_func)

            num_evals += 1

        return self.best_func

    def update(self, func, alpha, mu, tau):
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.mini = False
        self.best_fitness = np.inf

        # Re-initialize the search space
        self.dim = func.shape[1]
        self.temp = 1.0
        self.best_func = None

        # Update the individual lines of the selected solution
        for _ in range(100):
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
                self.best_fitness = func(new_func)
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation
                self.best_fitness = func(new_func)

            # Update the search space
            self.dim += 1
            self.temp *= 0.9

            # Check for convergence
            if self.best_fitness == func(self.best_func):
                self.mini = True
                break

        return self.best_func