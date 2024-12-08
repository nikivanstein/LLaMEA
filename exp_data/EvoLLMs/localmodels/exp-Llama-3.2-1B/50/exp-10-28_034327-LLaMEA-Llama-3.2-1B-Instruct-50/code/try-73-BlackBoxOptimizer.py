# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
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
        # Evolve the population using Iterated Permutation and Cooling Algorithm
        while self.func_evals < self.budget:
            # Select the fittest individuals
            fittest = sorted(self.population, key=self.func_evals, reverse=True)[:100]
            # Generate a new population by iterated permutation and cooling
            new_population = []
            for _ in range(100):
                # Select a random individual from the fittest population
                individual = fittest.pop()
                # Generate a new point using iterated permutation
                point = random.sample(individual, self.dim)
                # Evaluate the function at the point
                value = func(point)
                # Check if the point is within the bounds
                if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                    # If the point is within bounds, add it to the new population
                    new_population.append(point)
            # Add the new population to the existing population
            self.population.extend(new_population)
            # Update the function evaluations
            self.func_evals += len(new_population)
        # Return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# BlackBoxOptimizer: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# ```python
# ```python