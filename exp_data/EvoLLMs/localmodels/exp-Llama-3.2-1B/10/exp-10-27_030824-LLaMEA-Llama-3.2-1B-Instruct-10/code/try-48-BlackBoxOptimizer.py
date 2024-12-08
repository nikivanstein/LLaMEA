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

# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm combines random walk and linear interpolation to explore the search space.
# The algorithm starts at a random point and evaluates the function at each step, using linear interpolation to move towards the optimal solution.
# The probability of moving in the x-direction is 0.1, and the probability of moving in the y-direction is 0.9.
# This allows the algorithm to balance exploration and exploitation, and can lead to better solutions in many cases.

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Initialize the current point and evaluation
        current_point = np.random.uniform(self.search_space[0], self.search_space[1])
        current_evaluation = func(current_point)

        # Initialize the best point and evaluation
        best_point = current_point
        best_evaluation = current_evaluation

        # Initialize the step size for random walk
        step_size = 0.1

        # Initialize the step counter
        step_counter = 0

        # Loop until the budget is reached or the optimal solution is found
        while self.func_evaluations < self.budget:
            # Generate a new point using linear interpolation
            new_point = current_point + step_size * (current_evaluation - best_evaluation)

            # Evaluate the function at the new point
            new_evaluation = func(new_point)

            # Increment the function evaluations
            self.func_evaluations += 1

            # Update the best point and evaluation
            if new_evaluation > best_evaluation:
                best_point = new_point
                best_evaluation = new_evaluation

            # Update the current point and step size
            current_point = new_point
            step_size *= 0.9
            step_counter += 1

            # If the optimal solution is found, return the best point and evaluation
            if step_counter >= self.budget / 2:
                return best_point, best_evaluation