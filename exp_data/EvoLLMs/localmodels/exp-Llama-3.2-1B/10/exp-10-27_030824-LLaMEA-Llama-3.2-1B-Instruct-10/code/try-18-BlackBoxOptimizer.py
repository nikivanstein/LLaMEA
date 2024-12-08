import random
import numpy as np
from scipy.optimize import minimize_scalar

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def new_individual(self, evaluation, budget):
        # Apply linear interpolation to the evaluation
        point = evaluation + self.search_space[0] * (evaluation - self.search_space[0]) / (self.budget - self.func_evaluations)
        # If the budget is reached, return a default point and evaluation
        if budget <= self.func_evaluations:
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))
        else:
            # Refine the individual using random walk
            new_point = point + np.random.uniform(-1, 1, size=self.dim)
            return new_point, func(new_point)

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Updated BlackBoxOptimizer class
class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.budget = budget
        self.dim = dim

    def new_individual(self, evaluation, budget):
        # Apply linear interpolation to the evaluation
        point = evaluation + self.search_space[0] * (evaluation - self.search_space[0]) / (self.budget - self.func_evaluations)
        # If the budget is reached, return a default point and evaluation
        if budget <= self.func_evaluations:
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))
        else:
            # Refine the individual using random walk
            new_point = point + np.random.uniform(-1, 1, size=self.dim)
            return new_point, func(new_point)

# Usage
optimizer = BlackBoxOptimizer(1000, 5)
new_individual, evaluation = optimizer(new_individual, 1000)
print(f"New Individual: {new_individual}, Evaluation: {evaluation}")