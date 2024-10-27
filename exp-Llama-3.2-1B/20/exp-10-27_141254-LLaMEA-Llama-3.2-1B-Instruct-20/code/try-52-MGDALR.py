# Description: Adaptive Black Box Optimization using Genetic Algorithm and Evolution Strategies
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def select_strategy(self, func, budget):
        # Select a strategy based on the budget
        if budget < 100:
            # Use a simple strategy: always explore
            strategy = 'explore'
        else:
            # Use a more sophisticated strategy: adapt to the fitness landscape
            strategy = 'adapt'
        
        # Refine the strategy based on the fitness landscape
        if strategy == 'explore':
            # If the budget is small, always explore
            if self.explore_count < 100:
                return 'explore'
            else:
                # If the budget is large, adapt to the fitness landscape
                return 'adapt'
        elif strategy == 'adapt':
            # If the fitness landscape is flat, explore more
            if np.mean(np.abs(np.gradient(func(x))) < 1):
                return 'explore'
            else:
                # If the fitness landscape is not flat, adapt to the fitness landscape
                return 'adapt'
        else:
            raise ValueError('Invalid strategy')

    def optimize_func(self, func, budget):
        # Optimize the function using the selected strategy
        strategy = self.select_strategy(func, budget)
        if strategy == 'explore':
            # If the strategy is 'explore', explore the search space
            return self.__call__(func)
        elif strategy == 'adapt':
            # If the strategy is 'adapt', adapt to the fitness landscape
            return self.__call__(func)
        else:
            raise ValueError('Invalid strategy')

# Example usage:
if __name__ == '__main__':
    # Define the function to optimize
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an instance of the MGDALR algorithm
    mgdalr = MGDALR(100, 2)
    
    # Optimize the function using the MGDALR algorithm
    result = mgdalr.optimize_func(func, 100)
    print(result)