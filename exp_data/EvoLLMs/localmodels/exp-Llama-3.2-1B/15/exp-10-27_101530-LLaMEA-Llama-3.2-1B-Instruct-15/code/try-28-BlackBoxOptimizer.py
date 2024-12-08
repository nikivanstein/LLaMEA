import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_point = None
        self.best_score = -np.inf

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
        self.best_score = func(self.best_point)
        return self.best_point

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def __call__(self, func):
        best_point = self.optimizer(func)
        new_point = best_point + np.random.uniform(-0.1, 0.1, self.optimizer.dim)
        new_point = (new_point[0] + random.uniform(-0.1, 0.1), new_point[1] + random.uniform(-0.1, 0.1))
        if np.linalg.norm(new_point - best_point) < 0.01:
            return new_point
        return best_point

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 