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
        self mutation_prob = 0.25

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
            self.temp = min(1.0, self.temp + self.alpha * (1.0 - self.temp) * (num_evals / self.budget))

            num_evals += 1

        # Apply mutation
        if random.random() < self.mutation_prob:
            # Generate a random mutation
            mutation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the mutated function
            mutated_func = func + mutation

            # Check if the mutated function is better
            if np.random.rand() < self.alpha:
                self.best_func = mutated_func
            else:
                # If the mutated function is not better, revert the mutation
                mutated_func = func + mutation
                mutation *= self.tau

            # Update temperature
            self.temp = min(1.0, self.temp + self.alpha * (1.0 - self.temp) * (num_evals / self.budget))

        return self.best_func

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Mutation
# Code: 