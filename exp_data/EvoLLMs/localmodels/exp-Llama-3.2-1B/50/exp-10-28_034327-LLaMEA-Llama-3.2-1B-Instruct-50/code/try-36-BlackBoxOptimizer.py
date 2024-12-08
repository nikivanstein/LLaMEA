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

    def iterated_permutation(self, func, budget, dim, initial_population):
        population = initial_population
        best_individual = None
        best_value = -np.inf

        for _ in range(budget):
            # Select a random individual
            individual = random.choice(population)
            # Evaluate the function at the individual
            value = func(individual)
            # Check if the individual is within the bounds
            if -5.0 <= individual[0] <= 5.0 and -5.0 <= individual[1] <= 5.0:
                # If the individual is within bounds, update the best individual and value
                best_individual = individual
                best_value = value
            # Refine the strategy
            if random.random() < 0.45:
                # Change the direction of the search
                direction = np.random.uniform(-1, 1, self.dim)
                # Update the individual in the direction
                individual += direction
            else:
                # Change the size of the search space
                self.dim += 1
                # Generate a new individual
                individual = np.random.uniform(-5.0, 5.0, self.dim)

        # Return the best individual and value
        return best_individual, best_value

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# BlackBoxOptimizer: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 