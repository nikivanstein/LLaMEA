import numpy as np
from scipy.optimize import minimize
from collections import deque

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.explore_history = deque(maxlen=1000)

    def __call__(self, func, initial_individual=None):
        if initial_individual is None:
            initial_individual = np.array([-5.0] * self.dim)
        
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = initial_individual
        
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
        
        # Refine the strategy using probability 0.2
        if np.random.rand() < 0.2:
            # Randomly perturb the current x
            new_individual = x + np.random.normal(0, 0.1, self.dim)
        else:
            # Refine the current x using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(new_individual - inner(new_individual), np.gradient(y))
            new_individual += learning_rate * dx
        
        # Update the exploration history
        self.explore_history.append((x, y, new_individual))
        
        return new_individual

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using gradient descent and probability refinement.