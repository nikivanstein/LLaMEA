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
        for _ in range(iterations):
            # Refine the strategy using the probability 0.45
            if random.random() < 0.45:
                # Perturb the current individual
                new_x = [x + random.uniform(-0.01, 0.01) for x in initial_guess]
            else:
                # Refine the current individual
                new_x = initial_guess
            # Evaluate the new individual
            new_value = self.func(new_x)
            # Update the best individual
            if new_value < self.func(initial_guess):
                initial_guess = new_x
        return initial_guess, self.func(initial_guess)

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy