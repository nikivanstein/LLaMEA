import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_point = None
        self.best_fitness = -np.inf

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
        self.best_point = self.search_space[0], self.search_space[1]
        self.best_fitness = -np.inf
        # Refine the strategy
        if self.best_fitness == -np.inf:
            self.best_fitness = func(self.best_point)
            self.best_point = (self.best_point[0], self.best_point[1])
        return self.best_point

def bbo_suggest_point(func, search_space, budget):
    optimizer = BlackBoxOptimizer(budget, len(search_space))
    new_point = optimizer(func)
    return new_point

def bbo_suggest_point_refine(func, search_space, budget):
    optimizer = BlackBoxOptimizer(budget, len(search_space))
    while True:
        new_point = optimizer(func)
        func_value = func(new_point)
        if func_value < -np.inf:
            return new_point
        if func_value == -np.inf:
            return search_space[0], search_space[1]
        if new_point == optimizer.best_point:
            return optimizer.best_point

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 