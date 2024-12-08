import numpy as np
from scipy.optimize import minimize

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

    def optimize(self, func):
        # Initialize the solution with a random point in the search space
        solution = np.array([-5.0] * self.dim)
        
        # Run the optimization algorithm
        result = self.__call__(func, (solution,))
        
        # Refine the solution using the learned direction
        refined_solution = np.copy(solution)
        for _ in range(self.budget):
            # Evaluate the function at the current refined solution
            y = func(refined_solution)
            
            # If we've reached the maximum number of iterations, stop refining
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop refining
            if refined_solution[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(refined_solution - func(refined_solution), np.gradient(y))
            refined_solution += learning_rate * dx
        
        return refined_solution

# One-line description:
# Novel metaheuristic algorithm for black box optimization using gradient descent and refinement.

# Code: