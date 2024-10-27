# Description: Novel Metaheuristic Algorithm for Black Box Optimization (BBOB)
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func, initial_point=None, iterations=100, mutation_rate=0.01):
        # Initialize the current best point
        current_best_point = None
        current_best_fitness = float('-inf')

        # Generate an initial point
        if initial_point is None:
            initial_point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        # Evaluate the initial point
        current_best_fitness = func(initial_point)

        # Main loop
        for _ in range(iterations):
            # Generate a new point
            new_point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))

            # Evaluate the new point
            new_fitness = func(new_point)

            # Check if the new point is within the budget
            if new_fitness > current_best_fitness:
                # If not, update the current best point
                current_best_point = new_point
                current_best_fitness = new_fitness
            else:
                # If the new point is within the budget, check if it's better than the current best point
                if new_fitness > current_best_fitness + 0.0001:
                    # If it's better, update the current best point
                    current_best_point = new_point

            # Check if the budget is reached
            if new_fitness > current_best_fitness + 0.0001:
                # If not, return the current best point
                return current_best_point

        # If the budget is reached, return the current best point
        return current_best_point

# One-line description
# Novel Metaheuristic Algorithm for Black Box Optimization
# Optimizes black box functions using a novel metaheuristic algorithm