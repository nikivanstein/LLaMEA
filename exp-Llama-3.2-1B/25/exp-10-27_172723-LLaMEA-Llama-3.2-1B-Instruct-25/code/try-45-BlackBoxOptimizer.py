import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_score = float('inf')

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

    def select_new_individual(self, func, budget):
        # Select a new individual by changing the first line of the selected solution
        new_individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]

        # Evaluate the function at the new individual
        new_value = func(new_individual)

        # Check if the new individual is better than the best solution found so far
        if new_value < self.best_score:
            self.best_individual = new_individual
            self.best_score = new_value
            self.func_evaluations = 0
            # Update the budget
            self.budget = min(self.budget, budget)

        return new_individual

    def mutate(self, func):
        # Randomly change the first line of the selected solution
        new_individual = self.select_new_individual(func, self.budget)
        # Evaluate the new individual
        new_value = func(new_individual)
        # Check if the new individual is better than the best solution found so far
        if new_value < self.best_score:
            self.best_individual = new_individual
            self.best_score = new_value
            self.func_evaluations = 0
            # Update the budget
            self.budget = min(self.budget, self.budget + 1)
        return new_individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
# Code: 