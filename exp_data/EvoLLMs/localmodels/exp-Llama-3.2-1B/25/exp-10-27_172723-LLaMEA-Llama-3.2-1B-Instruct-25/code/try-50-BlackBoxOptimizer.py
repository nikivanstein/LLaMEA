import random
import numpy as np
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def _random_search(self, func, bounds, num_evaluations):
        # Perform random search
        for _ in range(num_evaluations):
            point = np.random.uniform(bounds[0], bounds[1])
            value = func(point)
            if value < 1e-10:  # arbitrary threshold
                return point

    def _mutation(self, func, bounds, point):
        # Perform mutation
        point = point + np.random.uniform(-math.sqrt(2), math.sqrt(2))
        value = func(point)
        if value < 1e-10:  # arbitrary threshold
            return point
        else:
            return point

    def __next_generation(self, func, bounds):
        # Perform mutation and random search
        generation = []
        while len(generation) < self.dim:
            generation.append(self._mutation(func, bounds, self._random_search(func, bounds, 100)))
        return generation

    def __next_solution(self, func, bounds):
        # Perform random search
        point = np.random.choice(bounds)
        value = func(point)
        if value < 1e-10:  # arbitrary threshold
            return point
        else:
            return point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"