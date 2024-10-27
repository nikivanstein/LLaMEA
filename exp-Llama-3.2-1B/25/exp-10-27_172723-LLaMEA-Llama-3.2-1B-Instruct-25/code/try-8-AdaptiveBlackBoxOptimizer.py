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

    def adaptive_search(self, func, budget, iterations):
        # Initialize the current solution
        current_solution = None

        # Initialize the current best solution
        current_best_solution = None

        # Initialize the current best fitness
        current_best_fitness = -np.inf

        # Iterate over the specified number of iterations
        for _ in range(iterations):
            # Evaluate the function for the specified number of times
            num_evaluations = min(budget, self.func_evaluations + 1)
            self.func_evaluations = self.func_evaluations + num_evaluations

            # Generate a random point in the search space
            point = np.random.choice(self.search_space)

            # Evaluate the function at the point
            value = func(point)

            # Check if the function has been evaluated within the budget
            if value < 1e-10:  # arbitrary threshold
                # If not, return the current point as the optimal solution
                return point
            else:
                # If the function has been evaluated within the budget, update the current solution
                if value > current_best_fitness:
                    current_best_solution = point
                    current_best_fitness = value
                # If the function has been evaluated within the budget, update the current solution
                elif value == current_best_fitness:
                    current_best_solution = np.array([current_best_solution])  # update the current best solution to a list

            # Update the current solution using the adaptive strategy
            if np.random.rand() < 0.25:
                current_solution = current_best_solution
            else:
                current_solution = point

        # Return the final solution
        return current_solution

# One-line description: "Adaptive Black Box Optimizer: An adaptive metaheuristic algorithm that adjusts its search strategy based on the performance of the current solution"
# Code: 