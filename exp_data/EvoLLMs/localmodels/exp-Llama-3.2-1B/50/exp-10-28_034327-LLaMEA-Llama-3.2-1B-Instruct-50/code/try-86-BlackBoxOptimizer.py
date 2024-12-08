import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

class IteratedPermutationCooling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []
        self.population_size = 100
        self.cooling_rate = 0.1
        self.iterations = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random permutation of the population
            permutation = random.sample(self.population, self.population_size)
            # Evaluate the function at each permutation
            values = [func(point) for point in permutation]
            # Calculate the average function value
            avg_value = np.mean(values)
            # Refine the strategy using the cooling rate
            if self.iterations < 100:
                self.iterations += 1
                # Select the best permutation based on the average function value
                best_permutation = permutation[np.argmax(values)]
                # Update the population with the best permutation
                self.population = [best_permutation]
            else:
                # Update the population with a new permutation
                self.population = permutation
            # Check if the budget is exceeded
            if self.func_evals >= self.budget:
                # Return the best point found so far
                return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 