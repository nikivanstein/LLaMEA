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
        self.population = None

    def __call__(self, func):
        # Initialize the population with random points in the search space
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Initialize the best point and its value
        best_point = None
        best_value = float('-inf')

        # Initialize the cooling schedule
        self.cooling_schedule = [100, 50, 20, 10, 5]

        # Iterate over the population
        for _ in range(self.budget):
            # Generate a new point in the search space
            new_point = self.population[np.random.randint(0, len(self.population))]

            # Evaluate the function at the new point
            value = func(new_point)

            # Check if the new point is within the bounds
            if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                # If the new point is within bounds, update the function value
                value = max(value, np.max(func(new_point)))

                # Check if the new point is better than the best point found so far
                if value > best_value:
                    # Update the best point and its value
                    best_point = new_point
                    best_value = value

                    # Refine the strategy using the new point
                    self.refine_strategy(best_point)

            # Update the population
            self.population = self.population[:]

            # Update the cooling schedule
            self.cooling_schedule = self.cooling_schedule[1:] + [self.cooling_schedule[0]]

        # Return the best point found
        return best_point

    def refine_strategy(self, best_point):
        # Generate a new point in the search space
        new_point = best_point + np.random.uniform(-1.0, 1.0, self.dim)

        # Evaluate the function at the new point
        value = func(new_point)

        # Check if the new point is within the bounds
        if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
            # If the new point is within bounds, update the function value
            value = max(value, np.max(func(new_point)))

        # Check if the new point is better than the best point found so far
        if value > self.best_value:
            # Update the best point and its value
            self.best_point = new_point
            self.best_value = value

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 