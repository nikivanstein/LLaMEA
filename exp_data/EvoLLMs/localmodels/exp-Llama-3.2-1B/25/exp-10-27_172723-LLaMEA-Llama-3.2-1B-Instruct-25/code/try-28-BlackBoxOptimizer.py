import random
import numpy as np

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

    def __next_solution(self, func, budget):
        # Refine the solution by changing a single line of the current solution
        new_point = point.copy()
        if np.random.rand() < 0.25:
            # Change a random line of the solution
            line_idx = random.randint(0, self.dim - 1)
            new_point[line_idx] = np.random.uniform(-5.0, 5.0)
        else:
            # Keep the solution as it is
            pass
        # Evaluate the function at the new point
        new_value = func(new_point)
        # Check if the new solution is better than the current solution
        if new_value > value:
            # If yes, return the new solution
            return new_point
        else:
            # If not, return the current solution
            return point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
# Code: 