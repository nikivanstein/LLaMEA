import numpy as np
import random
import copy
from collections import deque
from operator import add

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

            num_evals += 1

        return self.best_func

class NonLocalTemperatureMutationMetaheuristic(NonLocalTemperatureMetaheuristic):
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
                # Apply mutation with probability 1 - mu
                if random.random() < (1 - self.mu):
                    new_func = func + perturbation
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            num_evals += 1

        return self.best_func

class NonLocalTemperatureCrossoverMetaheuristic(NonLocalTemperatureMetaheuristic):
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        super().__init__(budget, dim, alpha, mu, tau)

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate two random parents
            parent1 = np.random.uniform(-self.dim, self.dim)
            parent2 = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + parent1 + parent2

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                # Apply crossover with probability 1 - mu
                if random.random() < (1 - self.mu):
                    new_func = func + parent1 + parent2
            else:
                # If the new function is not better, revert the parents
                parent1 *= self.tau
                parent2 *= self.tau
                new_func = func + parent1 + parent2

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 