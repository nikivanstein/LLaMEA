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

    def __str__(self):
        return f"Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm"

    def _cooling(self, current_value, best_value, temperature):
        if temperature == 0:
            return best_value
        else:
            return current_value - temperature * (best_value - current_value)

    def _iterated_permutation(self, current_value, best_value, temperature):
        # Generate a new population by iterating over all possible permutations of the current population
        new_population = []
        for _ in range(self.dim):
            # Generate a random permutation of the current population
            permutation = list(range(self.dim))
            random.shuffle(permutation)
            # Evaluate the function at the permutation
            value = func(permutation)
            # Check if the permutation is within the bounds
            if -5.0 <= permutation[0] <= 5.0 and -5.0 <= permutation[1] <= 5.0:
                # If the permutation is within bounds, update the function value
                new_value = value
                # Update the best value if necessary
                best_value = max(best_value, new_value)
                # Update the temperature
                temperature = min(temperature + 0.1, 1.0)
                # Add the permutation to the new population
                new_population.append(permutation)
        # Replace the current population with the new population
        self.func_evals = 0
        self.dim = len(new_population)
        self.func_evals = 0
        self.dim = len(new_population)
        self.func_evals = 0
        self.dim = len(new_population)
        self.func_evals = 0
        self.dim = len(new_population)
        return new_population

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 