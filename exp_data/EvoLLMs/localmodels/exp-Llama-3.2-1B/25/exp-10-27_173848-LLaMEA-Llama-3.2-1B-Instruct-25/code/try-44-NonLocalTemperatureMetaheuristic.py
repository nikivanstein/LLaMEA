import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None

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
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update temperature
            self.temp = min(1.0, self.temp * self.mu)

            num_evals += 1

        return self.best_func

class NonLocalTemperatureEvolutionaryMetaheuristic(NonLocalTemperatureMetaheuristic):
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        super().__init__(budget, dim, alpha, mu, tau)

    def __call__(self, func):
        # Refine strategy using non-local temperature
        updated_func = self.f(func, self.temp)
        self.temp = max(0.1, self.temp * self.mu)
        return updated_func

    def f(self, func, temp):
        # Evaluate the function with updated temperature
        return self.budget * func / temp

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 