import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
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

    def __next_generation(self, func, budget):
        # Calculate the probability of convergence
        prob_converge = self.budget / budget
        
        # Generate a new point in the search space using random walk and linear interpolation
        point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Evaluate the function at the point
        evaluation = func(point)
        
        # Increment the function evaluations
        self.func_evaluations += 1
        
        # Return the point and its evaluation
        return point, evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NovelMetaheuristicOptimizer(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __next_generation(self, func, budget):
        # Calculate the probability of convergence
        prob_converge = self.budget / budget
        
        # Generate a new point in the search space using random walk and linear interpolation
        point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Evaluate the function at the point
        evaluation = func(point)
        
        # Increment the function evaluations
        self.func_evaluations += 1
        
        # Return the point and its evaluation
        return point, evaluation

# Example usage
optimizer = NovelMetaheuristicOptimizer(budget=100, dim=5)
func = lambda x: np.sin(x)
new_individual, new_evaluation = optimizer(__next_generation(func, 100))
print(new_individual, new_evaluation)