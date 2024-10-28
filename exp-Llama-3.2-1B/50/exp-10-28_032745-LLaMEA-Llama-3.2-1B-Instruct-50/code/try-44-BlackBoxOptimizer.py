import numpy as np
from scipy.optimize import minimize
import random
import copy

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
                new_x = copy.deepcopy(best_x)
                for _ in range(100):  # Refine the strategy using probability 0.45
                    new_x[i] += random.uniform(-0.01, 0.01)
                    new_value = self.func(new_x)
                    if new_value < best_value:
                        best_x = new_x
                        best_value = new_value
                if random.random() < 0.45:  # Apply the probability to change the individual lines of the strategy
                    new_x[i] -= random.uniform(-0.01, 0.01)
            initial_guess = best_x
        return best_x, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy