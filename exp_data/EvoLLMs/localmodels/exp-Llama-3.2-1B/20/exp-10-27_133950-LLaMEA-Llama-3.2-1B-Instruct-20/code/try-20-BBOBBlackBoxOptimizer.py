# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.search_space_copy = self.search_space[:]

    def __call__(self, func, initial_individual=None, strategy=None):
        if initial_individual is None:
            initial_individual = self.search_space_copy[:]

        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, initial_individual, method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

        # Refine the strategy
        if strategy is None:
            strategy = self.refine_strategy(initial_individual, result.x)
        return strategy

    def refine_strategy(self, initial_individual, result_x):
        # Simple strategy: move towards the minimum/maximum of the function
        if result_x < initial_individual:
            return initial_individual - 0.1 * (initial_individual - result_x)
        else:
            return initial_individual + 0.1 * (initial_individual - result_x)

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)

# Run the optimization algorithm multiple times to evaluate the solution
for _ in range(10):
    result = optimizer(func, initial_individual=np.array([1.0]), strategy=None)
    print(result)