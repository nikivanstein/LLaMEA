import numpy as np
from scipy.optimize import minimize
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            # Introduce a probabilistic step to refine the strategy
            if random.random() < 0.2:
                # Randomly select a new direction for exploration
                new_direction = np.random.uniform(-5.0, 5.0, self.dim)
                # Refine the search space by adding a small random perturbation
                new_search_space = self.search_space + np.random.uniform(-0.1, 0.1, self.dim)
            else:
                # Use a deterministic step to converge to the optimal solution
                new_search_space = self.search_space
            # Refine the function evaluation to ensure the budget is not exceeded
            self.func_evaluations += 1
            try:
                result = minimize(wrapper, x=new_search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
                return result.x
            except Exception as e:
                print(f"Error: {e}")
                return None

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)