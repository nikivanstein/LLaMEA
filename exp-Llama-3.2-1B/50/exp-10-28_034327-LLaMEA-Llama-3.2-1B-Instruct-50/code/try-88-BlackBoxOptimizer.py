import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.best_point = None
        self.best_value = -np.inf

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

    def iterated_permutation(self, func, budget):
        while self.func_evals < budget:
            # Initialize the population with random points
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
            # Initialize the best point and value
            self.best_point = None
            self.best_value = -np.inf
            # Iterate through the population
            for _ in range(100):
                # Generate a new point using the iterated permutation
                point = np.random.permutation(population)
                # Evaluate the function at the point
                value = func(point)
                # Check if the point is within the bounds
                if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                    # If the point is within bounds, update the best point and value
                    self.best_point = point
                    self.best_value = value
                    # Update the population with the best point
                    population = [point if point == self.best_point else point for point in population]
                    # If the budget is exceeded, return the best point
                    if self.func_evals == budget:
                        return self.best_point
        # If the budget is exceeded, return the best point found
        return self.best_point

    def cooling_schedule(self, initial_value, cooling_rate):
        return initial_value / (1 + cooling_rate * (self.func_evals / self.budget))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 