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
            # Adaptive line search
            if self.dim > 1:
                d = np.linalg.norm(best_x - initial_guess)
                alpha = 0.1
                beta = 0.5
                best_x = initial_guess + alpha * d * new_x
            # Cyclic exploration
            if random.random() < 0.1:
                # Randomly select a new individual from the search space
                new_x = np.random.uniform(self.search_space[0], self.search_space[1], self.dim)
            else:
                # Explore the current search space
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
            new_value = self.func(new_x)
            if new_value < best_value:
                best_x = new_x
                best_value = new_value
            initial_guess = best_x
        return best_x, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using adaptive line search and cyclic exploration