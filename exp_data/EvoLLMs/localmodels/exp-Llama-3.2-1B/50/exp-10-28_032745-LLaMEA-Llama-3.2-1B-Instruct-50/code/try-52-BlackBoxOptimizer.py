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
        # Novel Metaheuristic Algorithm for Black Box Optimization
        # 
        # Refine the strategy using the following rules:
        # 1.  Start with a random initial guess and refine it using the 0.45 rule
        # 2.  If the budget is reached, stop and return the best individual found
        # 3.  Otherwise, generate new individuals by perturbing the current best individual
        # 4.  Evaluate the new individuals using the black box function
        # 5.  If the new individual is better, update the best individual
        # 6.  If the budget is reached, stop and return the best individual found

        best_x = initial_guess
        best_value = self.func(best_x)
        for _ in range(iterations):
            # Refine the initial guess using the 0.45 rule
            if random.random() < 0.45:
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
            else:
                new_x = best_x

            # Generate new individuals by perturbing the current best individual
            for i in range(self.dim):
                new_x[i] += random.uniform(-0.01, 0.01)
                new_x[i] = max(-5.0, min(new_x[i], 5.0))

            # Evaluate the new individuals using the black box function
            new_value = self.func(new_x)

            # If the new individual is better, update the best individual
            if new_value < best_value:
                best_x = new_x
                best_value = new_value

            # If the budget is reached, stop and return the best individual found
            if _ >= self.budget:
                break

        return best_x, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy