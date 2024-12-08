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
        if self.iterations == 0:
            self.best_point = point
            self.best_value = value
        else:
            # Refine the strategy using Iterated Permutation and Cooling Algorithm
            self.refine_strategy(point, value)
        return self.best_value

    def refine_strategy(self, point, value):
        # Select the best individual from the current population
        new_individual = self.evaluate_fitness(self.population)
        # Select the best point from the current best point
        new_point = self.best_point
        # Update the best point and value
        self.best_point = new_point
        self.best_value = value
        # Generate a new point in the search space
        new_point = np.random.uniform(-5.0, 5.0, self.dim)
        # Evaluate the function at the new point
        new_value = func(new_point)
        # Check if the new point is within the bounds
        if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
            # If the new point is within bounds, update the function value
            self.func_evals += 1
            return new_value
        # If the new point is not within bounds, return the best point found so far
        else:
            return self.best_value

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 