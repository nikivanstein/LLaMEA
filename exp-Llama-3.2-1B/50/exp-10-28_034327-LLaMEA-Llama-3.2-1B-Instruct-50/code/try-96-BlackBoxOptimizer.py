import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = -np.inf
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
        if self.iterations == 0:
            self.best_individual = point
            self.best_fitness = value
        else:
            # Refine the individual's strategy
            if self.best_fitness == -np.inf:
                self.best_individual = point
                self.best_fitness = value
            else:
                # Change the individual's lines to refine its strategy
                lines = self.best_individual
                for _ in range(10):
                    # Generate a new point in the search space
                    new_point = np.random.uniform(-5.0, 5.0, self.dim)
                    # Evaluate the function at the new point
                    new_value = func(new_point)
                    # Check if the new point is within the bounds
                    if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                        # If the new point is within bounds, update the function value
                        self.func_evals += 1
                        lines[0] = new_point[0]
                        lines[1] = new_point[1]
                        new_value = func(new_point)
                        if new_value > self.best_fitness:
                            self.best_individual = new_point
                            self.best_fitness = new_value
            # Update the best individual and its fitness
            self.best_individual = lines
            self.best_fitness = self.func_evals
            self.iterations += 1

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# BlackBoxOptimizer: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 