import numpy as np
from collections import deque
import random

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

    def select_strategy(self, budget):
        # Select a strategy based on the budget
        if budget < 10:  # 10 evaluations
            return "random"
        elif budget < 50:  # 50 evaluations
            return "random + adaptive"
        else:  # more than 50 evaluations
            return "adaptive"

    def adaptive_strategy(self, budget):
        # Adaptive strategy based on the budget
        if budget < 10:  # 10 evaluations
            return "random"
        elif budget < 20:  # 20 evaluations
            return "random + adaptive"
        else:  # more than 20 evaluations
            return "adaptive"

    def mutate(self, point, strategy):
        # Mutate the point based on the strategy
        if strategy == "random":
            return point
        elif strategy == "random + adaptive":
            # Adaptive mutation
            mutation_prob = 0.1
            if random.random() < mutation_prob:
                return np.random.uniform(-1, 1, self.dim)
            else:
                return point
        elif strategy == "adaptive":
            # Adaptive mutation
            mutation_prob = 0.05
            if random.random() < mutation_prob:
                return np.random.uniform(-1, 1, self.dim)
            else:
                return point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
# Code: 