import numpy as np
from scipy.optimize import minimize

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

    def iterated_permutation(self, func, bounds, initial, budget):
        # Initialize the population with random points in the search space
        population = [initial]
        for _ in range(budget):
            # Evaluate the function at each point and select the best one
            new_population = [func(point) for point in population]
            best_point = np.argmax(new_population)
            # Select a new point using the iterated permutation strategy
            new_point = initial + bounds[best_point] * np.random.uniform(-1, 1, self.dim)
            # Check if the new point is within the bounds
            if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                # If the new point is within bounds, add it to the population
                population.append(new_point)
        # Return the best point in the population
        return np.max(population)

    def iterated_cooling(self, func, bounds, initial, budget):
        # Initialize the population with random points in the search space
        population = [initial]
        for _ in range(budget):
            # Evaluate the function at each point and select the best one
            new_population = [func(point) for point in population]
            best_point = np.argmax(new_population)
            # Select a new point using the iterated cooling strategy
            new_point = initial + bounds[best_point] * np.random.uniform(-1, 1, self.dim)
            # Check if the new point is within the bounds
            if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                # If the new point is within bounds, add it to the population
                population.append(new_point)
        # Return the best point in the population
        return np.max(population)

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# ```