import numpy as np
import random

class AdaptiveNonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9, mu_adapt=0.05, tau_adapt=0.95):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.mu_adapt = mu_adapt
        self.tau_adapt = tau_adapt
        self.temp = 1.0
        self.best_func = None
        self.iteration = 0

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

            # Update mu and tau if necessary
            if self.iteration >= 10 and self.iteration % 10 == 0:
                self.mu_adapt = max(0.1, self.mu_adapt * 1.2)
                self.tau_adapt = max(0.9, self.tau_adapt * 0.95)

            self.iteration += 1

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Adaptive Mutation
# Code: 