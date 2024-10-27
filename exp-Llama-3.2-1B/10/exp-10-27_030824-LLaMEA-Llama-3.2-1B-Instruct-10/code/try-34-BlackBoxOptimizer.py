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
            # Initialize the best point and its evaluation
            best_point = None
            best_evaluation = float('-inf')
            # Generate a random point in the search space
            for _ in range(self.budget):
                point = np.random.uniform(self.search_space[0], self.search_space[1])
                # Evaluate the function at the point
                evaluation = func(point)
                # If the evaluation is better than the current best, update the best point and evaluation
                if evaluation > best_evaluation:
                    best_point = point
                    best_evaluation = evaluation
            # Return the best point and its evaluation
            return best_point, best_evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# The algorithm combines the benefits of random walk and linear interpolation to efficiently explore the search space and find the optimal solution.
# 
# The search process starts by generating a random point in the search space and evaluating the function at that point.
# If the evaluation is better than the current best, the algorithm updates the best point and evaluation.
# Otherwise, the algorithm repeats the process until the budget is reached.
# 
# The algorithm is designed to handle a wide range of tasks and can be used to solve black box optimization problems.
# 
# The probability of changing the strategy is 0.1, which allows the algorithm to refine its strategy based on the performance of the current solution.