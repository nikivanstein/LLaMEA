import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.population_size = 100

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

    def iterated_permutation_cooling(self, population):
        # Initialize the population with random points
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        # Initialize the current best point and value
        current_best = None
        current_value = -np.inf
        # Initialize the cooling schedule
        cooling_schedule = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # Iterate over the population
        for _ in range(self.population_size):
            # Generate a new individual
            new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the new individual
            value = self.func(new_individual)
            # Check if the new individual is better than the current best
            if value > current_value:
                # Update the current best point and value
                current_best = new_individual
                current_value = value
            # Update the population with the new individual
            population[_] = new_individual
            # Update the current best point and value based on the cooling schedule
            self.iterations += 1
            if self.iterations % len(cooling_schedule) == 0:
                # Update the current best point and value based on the cooling schedule
                current_best = np.argmax(population)
                current_value = np.max(population[current_best])
        # Return the current best point and value
        return current_best, current_value

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 