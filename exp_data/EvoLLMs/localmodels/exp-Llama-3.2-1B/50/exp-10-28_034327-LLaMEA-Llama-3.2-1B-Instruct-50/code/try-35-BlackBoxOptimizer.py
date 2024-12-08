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

# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
def iterated_permutation_cooling(func, bounds, initial_point, budget, dim):
    # Initialize the population with random points in the search space
    population = [initial_point + np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

    # Run the algorithm for the specified budget
    for _ in range(budget):
        # Select the fittest individual
        fittest_individual = population[np.argmax([func(i) for i in population])]

        # Generate a new point by iterated permutation
        new_point = fittest_individual + np.random.uniform(-0.1, 0.1, dim)

        # Check if the new point is within the bounds
        if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
            # If the new point is within bounds, update the fittest individual
            population[np.argmax([func(i) for i in population])] = new_point
            # Keep track of the fittest individual
            self.func_evals += 1
            # Apply cooling schedule
            if _ % 100 == 0:
                self.budget *= 0.9

    # Return the fittest individual
    return np.max(func(population))