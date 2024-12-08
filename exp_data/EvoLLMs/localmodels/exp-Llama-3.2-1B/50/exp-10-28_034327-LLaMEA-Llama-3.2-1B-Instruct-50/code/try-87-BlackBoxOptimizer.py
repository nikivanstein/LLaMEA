import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_point = None
        self.best_value = -np.inf
        self.iterations = 0

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

    def iterated_permutation(self, func):
        # Initialize the current point
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        # Initialize the population
        population = [current_point]
        # Initialize the best point
        self.best_point = None
        self.best_value = -np.inf
        # Iterate for a specified number of iterations
        for _ in range(self.iterations):
            # Generate a new population
            new_population = []
            for _ in range(len(population)):
                # Generate a new point using iterated permutation
                new_point = np.random.uniform(-5.0, 5.0, self.dim)
                # Evaluate the function at the new point
                value = func(new_point)
                # Check if the new point is within the bounds
                if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                    # If the new point is within bounds, add it to the new population
                    new_population.append(new_point)
            # Replace the current population with the new population
            population = new_population
            # Update the best point and value
            if np.max(func(np.random.uniform(-5.0, 5.0, self.dim))) > self.best_value:
                self.best_point = current_point
                self.best_value = np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))
        # Return the best point found
        return self.best_point

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 