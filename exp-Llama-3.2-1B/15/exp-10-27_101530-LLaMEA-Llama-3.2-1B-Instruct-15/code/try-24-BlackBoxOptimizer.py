import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.tau = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space using non-uniform sampling
            points = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.dim, self.tau))
            # Evaluate the function at the points
            func_values = func(points)
            # Sort the points by function value
            sorted_points = np.argsort(func_values)
            # Select the top points based on the budget
            top_points = sorted_points[:self.budget]
            # Get the top points
            top_points = points[top_points]
            # Evaluate the function at the top points
            func_values = func(top_points)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the budget is reached
            if self.func_evaluations < self.budget:
                # If not, return the top points
                return top_points
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization using Non-Uniform Sampling