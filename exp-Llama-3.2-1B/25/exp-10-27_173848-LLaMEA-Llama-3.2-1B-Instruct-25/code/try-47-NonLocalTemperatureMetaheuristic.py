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

class BBOBOptimizer:
    def __init__(self, budget, dim, nonlocal_temperature_metaheuristic, noiseless_functions, noiseless_random_functions):
        self.budget = budget
        self.dim = dim
        self.nonlocal_temperature_metaheuristic = nonlocal_temperature_metaheuristic
        self.noiseless_functions = noiseless_functions
        self.noiseless_random_functions = noiseless_random_functions

    def __call__(self, func, num_evals):
        # Evaluate the function using the provided number of evaluations
        if num_evals < self.budget:
            num_evals = self.budget

        # Initialize the best function and its score
        best_func = None
        best_score = -np.inf

        # Iterate over the noiseless functions
        for func in self.noiseless_functions:
            # Evaluate the function using the provided number of evaluations
            score = self.nonlocal_temperature_metaheuristic(func, num_evals)

            # Update the best function and its score if necessary
            if score > best_score:
                best_func = func
                best_score = score

        # Update the best function and its score using the provided number of evaluations
        self.nonlocal_temperature_metaheuristic(best_func, num_evals)

        return best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 