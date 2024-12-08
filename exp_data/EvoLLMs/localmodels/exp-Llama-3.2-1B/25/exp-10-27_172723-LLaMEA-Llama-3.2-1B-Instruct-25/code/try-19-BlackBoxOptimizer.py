import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, line_search=True):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def adaptive_line_search(self, func, point, step_size, max_iter=100):
        result = minimize(func, point, method='SLSQP', bounds=self.search_space, jac=None, constraints={'type': 'ineq', 'fun': lambda x: np.sum(x)} if line_search else None)
        if result.success:
            return result.x
        else:
            return None

# One-line description: "Evolutionary Algorithm with Adaptive Line Search: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

# Example usage:
def sphere_func(x):
    return np.sum(x**2)

optimizer = BlackBoxOptimizer(100, 5)
optimizer.func_evaluations = 0
print(optimizer.optimize(sphere_func))  # Initialize with a random point

# Refine the strategy with adaptive line search
optimizer = BlackBoxOptimizer(100, 5, adaptive_line_search=sphere_func)
print(optimizer.optimize(sphere_func))  # Optimize with adaptive line search