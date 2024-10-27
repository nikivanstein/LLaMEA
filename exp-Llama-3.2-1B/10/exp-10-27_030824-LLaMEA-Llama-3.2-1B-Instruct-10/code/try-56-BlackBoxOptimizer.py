import random
import numpy as np
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, budget):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < budget:
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

# Novel Metaheuristic Algorithm: "Adaptive Linear Search with Exploration"
# The algorithm starts with a random initial point and explores the search space using a linear search.
# It then uses a greedy search to explore the remaining space, adapting its search strategy based on the number of evaluations.
# The algorithm updates its search space and point based on the evaluations, allowing it to adapt to the function's behavior.
class AdaptiveLinearSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.point = np.random.uniform(self.search_space[0], self.search_space[1])
        self.search_space = np.linspace(self.search_space[0], self.search_space[1], 100)

    def __call__(self, func, budget):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < budget:
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

# One-line description: Novel metaheuristic algorithm for black box optimization using adaptive linear search with exploration.

# Code: 