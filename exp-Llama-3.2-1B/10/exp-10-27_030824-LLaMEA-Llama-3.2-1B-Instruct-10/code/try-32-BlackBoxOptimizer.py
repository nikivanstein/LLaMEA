import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def update_strategy(self, new_individual, func):
        # Define the new strategy
        if self.func_evaluations == 0:
            strategy = "random walk"
        elif self.func_evaluations < 10:
            strategy = "linear interpolation"
        else:
            strategy = "combining both"

        # Update the strategy based on the new individual's fitness
        if strategy == "random walk":
            # If the new individual's fitness is lower, switch to linear interpolation
            if func(new_individual) < func(new_individual + 0.1):
                strategy = "linear interpolation"
        elif strategy == "linear interpolation":
            # If the new individual's fitness is lower, switch to random walk
            if func(new_individual) < func(new_individual - 0.1):
                strategy = "random walk"

        # Update the individual's strategy
        new_individual.strategy = strategy

        # Update the individual's fitness
        new_individual.fitness = func(new_individual)

        # Increment the function evaluations
        self.func_evaluations += 1

        # Return the updated individual and its fitness
        return new_individual, func(new_individual)

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: