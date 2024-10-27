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
            
            # Learn a new direction using gradient descent and evolutionary strategy
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            if random.random() < 0.2:  # 20% chance of changing the individual
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            else:
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class GeneticProgramming:
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
            
            # Learn a new direction using gradient descent and evolutionary strategy
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            if random.random() < 0.2:  # 20% chance of changing the individual
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            else:
                dx = -np.dot(x - inner(x), np.gradient(y))
                x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

# Example usage
def func(x):
    return np.sum(x**2)

mgdalr = MGDALR(100, 10)
mgdalr_func = GeneticProgramming(100, 10)

mgdalr_func(mgdalr, func)