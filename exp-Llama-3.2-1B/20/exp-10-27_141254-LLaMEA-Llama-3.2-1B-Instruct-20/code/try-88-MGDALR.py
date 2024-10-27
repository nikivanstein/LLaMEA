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

    def select_strategy(self, x):
        # Probability of changing the individual to refine its strategy
        prob_change = 0.2
        
        # Change the individual to a new direction
        new_individual = np.array(x) + np.random.normal(0, 1, self.dim)
        
        # If the new individual is within the search space, refine its strategy
        if np.all(new_individual >= -5.0) and np.all(new_individual <= 5.0):
            return new_individual
        else:
            # Otherwise, return the original individual
            return x

# Test the algorithm
mgdalr = MGDALR(budget=100, dim=10)
func = lambda x: x**2
x = mgdalr(func)
print("Optimized solution:", x)