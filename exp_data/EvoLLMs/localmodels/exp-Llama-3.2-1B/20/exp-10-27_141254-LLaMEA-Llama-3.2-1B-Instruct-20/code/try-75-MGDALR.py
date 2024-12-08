import numpy as np
import random

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

    def mutate(self, func, x):
        # Randomly change a single element in x
        idx = random.randint(0, self.dim - 1)
        x[idx] = random.uniform(-5.0, 5.0)
        return x

# Description: Novel metaheuristic algorithm for black box optimization tasks
# Code: 
# ```python
# MGDALR: Novel metaheuristic algorithm for black box optimization tasks
# 
# References:
# 1. MGDALR: MGDALR: Novel metaheuristic algorithm for black box optimization tasks
# 2. BBOB test suite: BBOB test suite: Black Box Optimization Benchmarking
# 
# Time complexity: O(budget * dim^d), where d is the dimensionality and b is the budget
# Space complexity: O(budget * dim^d), where d is the dimensionality
# 
# Parameters:
#  - budget: Maximum number of function evaluations
#  - dim: Dimensionality of the search space
# 
# Notes:
#  - The algorithm uses gradient descent to optimize the function
#  - The exploration rate controls the trade-off between exploration and exploitation
#  - The mutation operator introduces random changes to the search space