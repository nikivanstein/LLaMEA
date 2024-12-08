import numpy as np
import random

class AdaptiveNonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9, mu_new=0.2, tau_new=0.8):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.mu_new = mu_new
        self.tau_new = tau_new
        self.temp = 1.0
        self.best_func = None
        self.best_fitness = np.inf

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_fitness > 0:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
                self.best_fitness = np.inf
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation
                # Update the mutation strategy
                if random.random() < self.mu_new:
                    self.mu_new *= 0.9
                    self.tau_new *= 0.99
                self.best_fitness = np.inf

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Mutation
# Code: 