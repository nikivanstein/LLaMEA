import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.step_size = 1.0
        self.adaptive_step_size = 0.0
        self.best_point = None

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
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def adapt_step_size(self):
        if self.func_evaluations == 0:
            return self.budget
        else:
            return min(self.budget, self.func_evaluations * 0.1 + self.budget * 0.9)

    def mutate(self, new_individual):
        if self.best_point is not None:
            # Calculate the step size for the mutation
            step_size = self.budget / (self.func_evaluations + 1)
            self.step_size = min(self.step_size, step_size)
            self.adaptive_step_size = max(self.adaptive_step_size, self.step_size)
            # Generate a random mutation
            new_point = (random.uniform(self.search_space[0] - self.step_size, self.search_space[1] + self.step_size), random.uniform(self.search_space[0] - self.step_size, self.search_space[1] + self.step_size))
            # Return the mutated point
            return new_point
        else:
            # Return the best point found so far
            return self.search_space[0], self.search_space[1]

    def evolve(self, new_individual):
        # Update the best point
        self.best_point = new_individual
        # Return the updated individual
        return new_individual