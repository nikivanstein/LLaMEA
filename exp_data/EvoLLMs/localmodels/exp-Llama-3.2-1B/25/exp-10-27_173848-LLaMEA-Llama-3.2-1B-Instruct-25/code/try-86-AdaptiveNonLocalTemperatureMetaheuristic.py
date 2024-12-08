import numpy as np
import random

class AdaptiveNonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9, mu_new=0.2):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.mu_new = mu_new
        self.temp = 1.0
        self.best_func = None
        self.population = None

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

            num_evals += 1

            # Update the population using adaptive mutation
            self.population = [self.evaluate_fitness(individual) for individual in self.population]
            self.population = [self.mu * individual + self.mu_new * (1 - self.mu) * (individual - self.mu) for individual in self.population]
            self.temp *= self.tau

        return self.best_func

    def evaluate_fitness(self, func):
        return np.random.uniform(-self.dim, self.dim) * func

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Mutation
# Code: 