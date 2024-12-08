import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
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

class IteratedPermutationCooling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.population = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random permutation of the current population
            permutation = np.random.permutation(self.population)
            # Evaluate the function at each point in the permutation
            values = [func(point) for point in permutation]
            # Calculate the average value of the function at the points
            average_value = np.mean(values)
            # Update the current population with the points that are within the bounds
            self.population = [point for point in permutation if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0]
            # Update the function value using the average value
            self.func_evals += 1
            self.population = np.random.permutation(self.population)
            # Apply cooling
            self.iterations += 1
            if self.iterations > 100:
                break
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 