import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.fitness_history = []

    def __call__(self, func, iterations):
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
            # Refine the strategy using probability 0.15
            if random.random() < 0.15:
                # Generate a new point with a different direction
                new_point = (point[0] + random.uniform(-1.0, 1.0), point[1] + random.uniform(-1.0, 1.0))
                # Check if the new point is within the budget
                if self.func_evaluations < self.budget:
                    # If not, return the new point
                    return new_point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

class MutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func, iterations):
        while self.func_evaluations < self.budget:
            # Generate a new individual
            new_individual = func(self.search_space[0], self.search_space[1])
            # Evaluate the new individual
            new_func_value = func(new_individual[0], new_individual[1])
            # Refine the strategy using probability 0.15
            if random.random() < 0.15:
                # Generate a new direction
                new_direction = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
                # Generate a new point using the new direction
                new_point = (new_individual[0] + new_direction[0]*new_individual[0], new_individual[1] + new_direction[1]*new_individual[1])
                # Check if the new point is within the budget
                if self.func_evaluations < self.budget:
                    # If not, return the new point
                    return new_point
            # Return the new individual
            return new_individual

# One-line description
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the strategy using probability 0.15
# 
# Code: 