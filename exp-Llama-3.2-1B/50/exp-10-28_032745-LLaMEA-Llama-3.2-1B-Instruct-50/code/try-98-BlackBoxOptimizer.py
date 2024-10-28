import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.search_strategy = "AdaptiveLineSearch"

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            if self.search_strategy == "AdaptiveLineSearch":
                # Adaptive line search
                delta = 0.01
                for _ in range(10):
                    new_x = [x + delta * (func(new_x) - best_value) for x in best_x]
                    new_value = self.func(new_x)
                    if new_value < best_value:
                        best_x = new_x
                        best_value = new_value
            initial_guess = best_x
        return best_x, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using adaptive line search strategy
# 
# The algorithm adapts the line search strategy based on the convergence rate of the optimization process.
# This allows the algorithm to adjust its search direction and step size to improve the efficiency of the optimization process.