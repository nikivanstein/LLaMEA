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

        # Reconstruct the new individual
        new_individual = copy.deepcopy(func)
        for i in range(self.dim):
            new_individual[i] += perturbation[i]

        # Evaluate the new individual
        new_func = self.evaluate_fitness(new_individual)

        return new_func

    def evaluate_fitness(self, func):
        # Reconstruct the problem
        problem = ioh.iohcpp.Sphere(func, self.dim)

        # Evaluate the new function
        new_func = problem.evaluate()

        # Return the score
        return new_func.score()

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# Description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python