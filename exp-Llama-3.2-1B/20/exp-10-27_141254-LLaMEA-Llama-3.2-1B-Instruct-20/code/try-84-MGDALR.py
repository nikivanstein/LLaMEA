# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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

    def adapt_strategy(self, individual):
        # Refine the strategy by changing the individual lines
        if random.random() < 0.2:
            # Increase exploration rate
            self.explore_rate *= 1.1
            # Decrease learning rate
            self.learning_rate *= 0.9
        
        elif random.random() < 0.1:
            # Decrease exploration rate
            self.explore_rate *= 0.9
            # Increase learning rate
            self.learning_rate *= 1.1
        
        return individual

# Test the algorithm
mgdalr = MGDALR(100, 10)
func = lambda x: x**2
individual = np.array([-5.0] * 10)
mgdalr(individual)