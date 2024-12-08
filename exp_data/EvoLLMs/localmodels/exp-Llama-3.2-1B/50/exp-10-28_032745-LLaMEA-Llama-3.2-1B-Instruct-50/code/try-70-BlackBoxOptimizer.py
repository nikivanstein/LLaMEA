import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

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
            initial_guess = best_x
        return best_x, best_value

    def novel_search(self, initial_guess, iterations):
        # Novel search strategy: Refine individual lines of the selected solution
        for _ in range(iterations):
            if _ % 10 == 0:  # Refine every 10 iterations
                for i in range(self.dim):
                    new_x = [x + random.uniform(-0.01, 0.01) for x in initial_guess]
                    new_value = self.func(new_x)
                    if new_value < self.func(initial_guess):
                        initial_guess = new_x
        return initial_guess

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Novel Metaheuristic Algorithm for Black Box Optimization
# 