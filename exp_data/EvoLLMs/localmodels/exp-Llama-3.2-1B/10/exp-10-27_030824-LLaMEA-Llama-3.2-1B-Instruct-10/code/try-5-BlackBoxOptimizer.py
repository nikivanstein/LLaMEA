import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_point, iterations=1000):
        # Initialize the current point and its evaluation
        current_point = initial_point
        current_evaluation = func(current_point)

        # Define the bounds for the current point
        bounds = [self.search_space[0], self.search_space[1]]

        # Perform a random walk for a specified number of iterations
        for _ in range(iterations):
            # Generate a new point using linear interpolation
            new_point = current_point + np.random.uniform(-bounds[1], bounds[1])
            # Evaluate the function at the new point
            new_evaluation = func(new_point)

            # If the budget is reached, return the current point and evaluation
            if self.func_evaluations < self.budget:
                return current_point, current_evaluation
            # Otherwise, update the current point and evaluation
            else:
                current_point = new_point
                current_evaluation = new_evaluation

        # If the budget is not reached, return the final point and evaluation
        return current_point, current_evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np
# from scipy.optimize import minimize

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_point, iterations=1000):
        # Initialize the current point and its evaluation
        current_point = initial_point
        current_evaluation = func(current_point)

        # Define the bounds for the current point
        bounds = [self.search_space[0], self.search_space[1]]

        # Perform a random walk for a specified number of iterations
        for _ in range(iterations):
            # Generate a new point using linear interpolation
            new_point = current_point + np.random.uniform(-bounds[1], bounds[1])
            # Evaluate the function at the new point
            new_evaluation = func(new_point)

            # If the budget is reached, return the current point and evaluation
            if self.func_evaluations < self.budget:
                return current_point, new_evaluation
            # Otherwise, update the current point and evaluation
            else:
                current_point = new_point
                current_evaluation = new_evaluation

        # If the budget is not reached, return the final point and evaluation
        return current_point, current_evaluation

# Usage
if __name__ == "__main__":
    # Create a problem instance
    problem = RealSingleObjectiveProblem(1, 2, 3)

    # Initialize the optimizer
    optimizer = NovelMetaheuristicOptimizer(1000, 3)

    # Call the optimizer
    new_individual, fitness = optimizer(problem, initial_point=[0.0, 0.0], iterations=1000)
    print("New Individual:", new_individual)
    print("Fitness:", fitness)