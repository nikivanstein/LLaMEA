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
            # Refine the strategy by changing the number of lines
            # of the selected solution to refine its strategy
            if self.func_evaluations % 100 == 0:
                self.func_evaluations += 1
                if self.func_evaluations < self.budget:
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

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the strategy by changing the number of lines
# of the selected solution to refine its strategy
# 
class NovelMetaheuristicOptimizer(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
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

# Test the code
optimizer = NovelMetaheuristicOptimizer(1000, 10)
func = lambda x: x**2
best_point = optimizer(func)
print(best_point)