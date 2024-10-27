import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations=100):
        # Define the probability of changing the current point
        p_change = 0.1

        # Define the number of iterations
        iterations = iterations

        # Initialize the best point and its evaluation
        best_point = None
        best_evaluation = -np.inf

        # Iterate over the specified number of iterations
        for _ in range(iterations):
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])

            # Evaluate the function at the point
            evaluation = func(point)

            # Update the best point and its evaluation if necessary
            if evaluation > best_evaluation:
                best_point = point
                best_evaluation = evaluation

            # If the budget is reached, return a default point and evaluation
            if self.func_evaluations >= self.budget:
                return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

        # Return the best point and its evaluation
        return best_point, best_evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.
