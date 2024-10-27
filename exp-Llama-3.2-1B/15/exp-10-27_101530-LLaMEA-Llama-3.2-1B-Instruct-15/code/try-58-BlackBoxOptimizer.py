import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.perturbations = 10

    def __call__(self, func):
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
            # If the point is out of bounds, perturb it
            if random.random() < 0.15:
                new_point = (point[0] + random.uniform(-self.perturbations, self.perturbations), point[1] + random.uniform(-self.perturbations, self.perturbations))
                # Check if the new point is within the budget
                if self.func_evaluations < self.budget:
                    # If not, return the new point
                    return new_point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 