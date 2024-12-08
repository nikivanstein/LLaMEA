import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = np.inf

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            
            # Refine the strategy
            if self.func_evaluations % 100 == 0 and self.func_evaluations < self.budget:
                # If the budget is reached, return the point
                return point
            elif self.func_evaluations % 100 == 0 and self.func_evaluations >= self.budget:
                # If the budget is reached, return the best point found so far
                return self.search_space[0], self.search_space[1]
            else:
                # If the budget is not reached, return the point
                return point
        
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 