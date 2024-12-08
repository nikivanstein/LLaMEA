import random
import numpy as np
from copy import deepcopy

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, adaptive_threshold=0.25):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.adaptive_threshold = adaptive_threshold
        self.best_solution = None
        self.best_fitness = float('inf')
        self.current_solution = None
        self.current_fitness = float('inf')

    def __call__(self, func, num_evaluations):
        if num_evaluations > self.func_evaluations + 1:
            num_evaluations = self.func_evaluations + 1

        # Evaluate the function for the specified number of times
        self.func_evaluations += num_evaluations
        func_evaluations = self.func_evaluations

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

    def optimize(self, func):
        # Initialize the current solution and fitness
        self.current_solution = None
        self.current_fitness = float('inf')
        self.best_solution = None
        self.best_fitness = float('inf')

        # Perform the optimization
        while self.func_evaluations < self.budget:
            # Evaluate the function for the specified number of times
            num_evaluations = min(self.budget, self.func_evaluations + 1)
            self.func_evaluations += num_evaluations

            # Generate a random point in the search space
            point = np.random.choice(self.search_space)

            # Evaluate the function at the point
            value = func(point)

            # Check if the function has been evaluated within the budget
            if value < 1e-10:  # arbitrary threshold
                # If not, update the current solution
                self.current_solution = point
                self.current_fitness = value
            else:
                # If the function has been evaluated within the budget, update the best solution if necessary
                if value < self.current_fitness:
                    self.best_solution = point
                    self.best_fitness = value
                # Update the current solution and fitness
                self.current_solution = point
                self.current_fitness = value

# One-line description: "Adaptive Black Box Optimizer: A novel metaheuristic algorithm that adapts its search strategy based on the results of previous evaluations"

# Example usage:
optimizer = AdaptiveBlackBoxOptimizer(budget=100, dim=5)
optimizer.optimize(lambda x: np.sum(x**2))
print(optimizer.best_solution)
print(optimizer.best_fitness)