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
        self.perturbations = None

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

            # Store the perturbation for later use
            self.perturbations.append(perturbation)

            num_evals += 1

        # Reconstruct the best function
        self.best_func = self.reconstruct_best_func()

        return self.best_func

    def reconstruct_best_func(self):
        if self.best_func is None:
            return None

        # Get the perturbations
        perturbations = self.perturbations[:self.budget]

        # Initialize the new function
        new_func = None

        # Iterate over the perturbations
        for perturbation in perturbations:
            # Evaluate the new function
            new_func = self.f(self.best_func, perturbation)

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = self.best_func + perturbation

        return new_func

    def f(self, func, perturbation):
        # Evaluate the new function
        return func + perturbation

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 