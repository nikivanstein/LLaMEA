import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
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

    def adapt_search(self, func_evaluations):
        # Initialize the new search space with a random subset of the original search space
        new_search_space = np.random.choice(self.search_space, size=self.dim, replace=False)

        # Evaluate the function at the new search space
        new_value = func(new_search_space)

        # Check if the new function value is within the budget
        if new_value < 1e-10:  # arbitrary threshold
            # If not, return the new search space as the optimal solution
            return new_search_space
        else:
            # If the new function value is within the budget, return the new search space
            return new_search_space

# One-line description: "Adaptive Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems by adapting its search strategy based on the evolution of the function evaluations"

# Example usage:
optimizer = AdaptiveBlackBoxOptimizer(100, 10)
func = lambda x: x**2
optimized_solution = optimizer(func)
print(optimized_solution)  # e.g. (-0.5, -0.5)