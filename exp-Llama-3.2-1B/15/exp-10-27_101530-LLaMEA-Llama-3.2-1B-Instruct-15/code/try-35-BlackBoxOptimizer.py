import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

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
        return self.search_space[0], self.search_space[1]

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.lines = [
            # "Line 1: Refine the search space to [-3.0, 3.0]"
            f"Line 1: Refine the search space to [{self.search_space[0]}, {self.search_space[1]}]",
            # "Line 2: Increase the budget to 1000"
            f"Line 2: Increase the budget to {self.budget}",
            # "Line 3: Use a more efficient evaluation function"
            f"Line 3: Use a more efficient evaluation function"
        ]

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
        return self.search_space[0], self.search_space[1]

# Initialize the optimizer with a budget of 1000 and a dimension of 5
optimizer = NovelMetaheuristicOptimizer(budget=1000, dim=5)

# Evaluate the function using the optimizer
func = lambda point: point[0]**2 + point[1]**2
best_point = optimizer(func)
print(best_point)