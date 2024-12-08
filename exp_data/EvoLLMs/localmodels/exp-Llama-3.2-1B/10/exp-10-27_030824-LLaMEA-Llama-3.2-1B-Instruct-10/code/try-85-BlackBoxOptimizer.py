# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.search_space_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]), func(np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]))

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Initialize the population with random points
            population = [np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]), func(np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]))]
            # Evaluate the population
            for _ in range(self.budget - 1):
                # Select the fittest individual
                fittest = population.index(max(population))
                # Generate a new individual by linear interpolation
                new_individual = [population[fittest][0], population[fittest][1] + (population[fittest + 1][0] - population[fittest][0]) * (np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]) - population[fittest][1]) / (self.budget - 1)]
                # Add the new individual to the population
                population.append(new_individual)
            # Return the fittest individual
            return population[0], population[0]
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]), func(np.random.uniform(self.search_space_bounds['x'][0], self.search_space_bounds['x'][1]))

# Initialize the Black Box Optimizer
optimizer = BlackBoxOptimizer(100, 10)

# Evaluate the black box function for 24 noiseless functions
results = []
for func in [eval("x**2 + y**2") for _ in range(24)]:
    result = optimizer(func)
    results.append(result)

# Print the results
print("Results:")
for i, result in enumerate(results):
    print(f"Function: {eval('x**2 + y**2')}, Optimization Algorithm: {optimizer.__class__.__name__}, Score: {result[1]}")