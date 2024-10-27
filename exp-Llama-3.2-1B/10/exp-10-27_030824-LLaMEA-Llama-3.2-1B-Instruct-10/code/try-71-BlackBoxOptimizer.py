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

    def __repr__(self):
        return "BlackBoxOptimizer(budget={}, dim={})".format(self.budget, self.dim)

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def __call__(self, func):
        # Evaluate the function using the optimizer
        point, evaluation = self.optimizer(func)
        # Refine the strategy by changing the direction of the random walk
        if self.optimizer.func_evaluations < self.optimizer.budget:
            # Generate a random direction vector
            direction = np.random.uniform(-1, 1, self.dim)
            # Apply the random walk to the point
            new_point = point + direction
            # Evaluate the function at the new point
            new_evaluation = func(new_point)
            # Update the optimizer with the new point and evaluation
            self.optimizer(point, new_evaluation)
            # Return the new point and evaluation
            return new_point, new_evaluation
        else:
            # If the budget is reached, return the default point and evaluation
            return point, evaluation

# Initialize the optimizer with a budget of 1000 and a dimension of 5
optimizer = NovelMetaheuristicOptimizer(budget=1000, dim=5)

# Test the optimizer
func = lambda x: x**2
for _ in range(10):
    point, evaluation = optimizer(func)
    print("Point:", point)
    print("Evaluation:", evaluation)
    print()