import numpy as np
import random

class Nova:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_point = None
        self.best_value = -np.inf

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

    def mutate(self, individual):
        # Refine the strategy by changing the individual's lines
        lines = individual.split('\n')
        # Select 3 random lines of the selected solution
        lines = lines[:3]
        # Change the lines to refine the strategy
        lines = [f'new_line_{i+1} = {random.uniform(-5.0, 5.0)}' for i in range(3)]
        # Join the lines back into a single string
        individual = '\n'.join(lines)
        return individual

    def evaluate_fitness(self, individual):
        # Refine the strategy by changing the individual's lines
        lines = individual.split('\n')
        # Select 3 random lines of the selected solution
        lines = lines[:3]
        # Change the lines to refine the strategy
        lines = [f'new_line_{i+1} = {random.uniform(-5.0, 5.0)}' for i in range(3)]
        # Join the lines back into a single string
        individual = '\n'.join(lines)
        # Evaluate the function at the new individual
        func_value = eval(individual)
        return func_value

# Initialize the Nova optimizer
nova = Nova(1000, 10)