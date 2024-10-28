import random
import numpy as np
from scipy.optimize import minimize
from typing import List

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

    def mutate(self, individual):
        # Randomly select two random indices in the individual
        idx1, idx2 = random.sample(range(self.dim), 2)

        # Swap the values at the two indices
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        # Evaluate the individual at the new point
        value = self.func(individual)

        # Check if the individual has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the new individual as the mutated solution
            return individual
        else:
            # If the individual has been evaluated within the budget, return the individual
            return individual

# One-line description: "Black Box Optimizer: A metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"