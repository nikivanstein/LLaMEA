import numpy as np
import random
import copy

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []

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

            # Create a new individual
            new_individual = copy.deepcopy(func)
            new_individual.perturbations.append(perturbation)

            num_evals += 1

        return self.best_func

    def __next__(self):
        # Select the best individual
        best_func = self.best_func
        best_individual = copy.deepcopy(best_func)

        # Refine the strategy
        if random.random() < 0.25:
            # Apply mutation
            mutation = np.random.uniform(-self.dim, self.dim)
            new_individual = copy.deepcopy(best_func)
            new_individual.mutations.append(mutation)

            # Update the best individual
            best_func = copy.deepcopy(best_individual)
            best_individual.perturbations.append(best_func.mutations[-1])

        return best_func, best_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 