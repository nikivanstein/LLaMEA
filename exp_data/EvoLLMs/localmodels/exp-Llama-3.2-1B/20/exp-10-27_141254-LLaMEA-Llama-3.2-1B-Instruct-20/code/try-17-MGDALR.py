# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
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

def objective(individual, func, budget):
    return func(individual)

def differential_evolution_bbo(func, bounds, budget):
    result = differential_evolution(objective, bounds, args=(func, budget))
    return result.x

def differential_evolution_mgdalr(func, bounds, budget):
    return MGDALR(budget, len(bounds)).__call__(func)

# Example usage:
bounds = [(None, None), (-5.0, 5.0)]  # 2D search space with bounds
budget = 100  # Number of function evaluations
func = lambda x: x[0]**2 + x[1]**2  # Simple 2D black box function

result = differential_evolution_mgdalr(func, bounds, budget)
print(result)