# Code:
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

    def select_strategy(self, x):
        # Define the selection function based on the current strategy
        if x[-1] < 5.0:
            # Select the individual with the highest fitness value
            return random.choice([i for i, f in enumerate(x) if f == np.max(x)])
        else:
            # Select the individual with the lowest fitness value
            return random.choice([i for i, f in enumerate(x) if f == np.min(x)])

    def refine_strategy(self, x, strategy):
        # Define the refinement function based on the selected strategy
        if strategy == 'exploration':
            # Use exploration-exploitation trade-off
            learning_rate = self.learning_rate * (1 - self.explore_rate / 10)
        elif strategy == 'exploitation':
            # Use exploitation strategy
            learning_rate = self.learning_rate
        else:
            raise ValueError("Invalid strategy. Choose 'exploration' or 'exploitation'.")

        # Learn a new direction using gradient descent
        dx = -np.dot(x - self.select_strategy(x), np.gradient(self.select_strategy(x)))
        x += learning_rate * dx

        return x

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using gradient descent and exploration-exploitation trade-off strategy.