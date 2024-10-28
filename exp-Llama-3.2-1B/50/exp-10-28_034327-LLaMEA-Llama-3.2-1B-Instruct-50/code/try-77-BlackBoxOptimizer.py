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

class IteratedPermutationCooling(BackBoxOptimizer):
    def __init__(self, budget, dim, initial_temperature, cooling_rate, mutation_rate):
        super().__init__(budget, dim)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.mutation_rate = mutation_rate
        self.current_temperature = self.initial_temperature

    def __call__(self, func):
        # Initialize the current population with random permutations
        population = np.random.permutation(self.dim * self.budget)

        for _ in range(self.dim):
            # Iterate through the population
            for i in range(self.dim):
                # Generate a new point by swapping two random points
                new_point = np.copy(population[i:i + self.dim])
                new_point[i], new_point[i + self.dim] = new_point[i + self.dim], new_point[i]
                # Evaluate the function at the new point
                value = func(new_point)
                # Check if the point is within the bounds
                if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                    # If the point is within bounds, update the function value
                    self.func_evals += 1
                    population[i:i + self.dim] = new_point
                    # Update the current temperature
                    self.current_temperature = min(self.current_temperature * self.cooling_rate, self.initial_temperature + self.mutation_rate * (self.current_temperature - self.initial_temperature))

        # Return the best point found
        return np.max(func(population))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 