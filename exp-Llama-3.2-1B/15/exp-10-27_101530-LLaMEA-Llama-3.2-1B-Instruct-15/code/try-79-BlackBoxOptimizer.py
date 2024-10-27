import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_point = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        if self.best_point is None:
            self.best_point = self.search_space[0], self.search_space[1]
        return self.best_point

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def __call__(self, func, num_iterations=100):
        for _ in range(num_iterations):
            # Refine the strategy by changing the number of evaluations
            self.optimizer.func_evaluations = 10 * _ / 5
            # Generate a new point with refined strategy
            new_point = self.optimizer.optimizer.__call__(func)
            # Evaluate the function at the new point
            func_value = func(new_point)
            # Update the best point found so far
            self.optimizer.best_point = new_point
            # Check if the budget is reached
            if self.optimizer.func_evaluations >= self.optimizer.budget:
                # If not, return the best point found so far
                return self.optimizer.best_point
        # If the budget is not reached after all iterations, return None
        return None

# Example usage:
from blackbox import bbo
bbo = bbo.BBOB()
optimizer = NovelMetaheuristicOptimizer(bbo.dim)
print(optimizer(bbo.func, num_iterations=10))