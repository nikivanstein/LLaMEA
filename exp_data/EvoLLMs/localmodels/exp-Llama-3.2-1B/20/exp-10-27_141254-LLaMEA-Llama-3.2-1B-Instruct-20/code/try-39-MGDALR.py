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

    def __call__(self, func, num_evals=100):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(num_evals):
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

def bbopt(func, budget, dim, num_evals=100):
    return MGDALR(budget, dim)(func, num_evals)

# Evaluate the black box function on the BBOB test suite
def evaluate_bbopt(func, budget, dim):
    return bbopt(func, budget, dim)

# Run the algorithm and print the result
result = evaluate_bbopt(lambda x: x**2, 100, 2)
print(f"Optimal solution: {result}")
print(f"Optimal fitness: {result**2}")