# Description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
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

            num_evals += 1

        return self.best_func

    def optimize(self, func):
        # Evaluate the function with the given budget
        self.best_func = self.__call__(func)

        # Refine the strategy using Non-Local Temperature and Adaptive Mutation
        for _ in range(self.budget):
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                # If the new function is better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation
            else:
                # If the new function is not better, add it to the population
                self.population.append(new_func)

        # Select the best individual
        self.population = self.population[:1]

        # Normalize the population
        self.population = np.array(self.population) / np.sum(self.population)

        # Update the best function
        self.best_func = self.population[np.argmax(self.population)]

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 