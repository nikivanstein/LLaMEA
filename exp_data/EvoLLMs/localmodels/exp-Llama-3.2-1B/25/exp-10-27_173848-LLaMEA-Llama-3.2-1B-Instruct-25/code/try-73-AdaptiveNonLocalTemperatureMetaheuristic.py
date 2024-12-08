import numpy as np
import random
import copy

class AdaptiveNonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.sigma = sigma
        self.temp = 1.0
        self.best_func = None
        self mutation_rate = 0.01

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

            # Adapt the temperature
            self.temp *= (1 - self.alpha * (num_evals / self.budget))

            num_evals += 1

        # Apply mutation
        if random.random() < self.mutation_rate:
            perturbation = np.random.uniform(-self.dim, self.dim)
            new_func = func + perturbation
            if np.random.rand() < self.alpha:
                new_func = func + perturbation
            else:
                perturbation *= self.tau
                new_func = func + perturbation

        return new_func

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Mutation
# Code: 