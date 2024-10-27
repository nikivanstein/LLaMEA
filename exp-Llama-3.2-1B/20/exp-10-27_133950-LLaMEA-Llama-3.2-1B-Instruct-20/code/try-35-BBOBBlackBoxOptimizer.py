# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

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
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def refine_strategy(self, new_individual, old_individual):
        # Probability of changing individual: 0.2
        if np.random.rand() < 0.2:
            # Refine the strategy by changing the individual's bounds
            new_bounds = (old_individual + 0.5, old_individual + 1.5)
            return new_individual, new_bounds

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)

# Refine the strategy
new_individual, new_bounds = result
new_individual, new_bounds = optimizer.refine_strategy(new_individual, result)
print(f"Refined individual: {new_individual}, Refined bounds: {new_bounds}")

# Example usage (continued)
optimizer = BBOBBlackBoxOptimizer(1000, 10)
result = optimizer(func)

# Refine the strategy (continued)
new_individual, new_bounds = result
new_individual, new_bounds = optimizer.refine_strategy(new_individual, result)
print(f"Refined individual: {new_individual}, Refined bounds: {new_bounds}")

# Example usage (continued)
optimizer = BBOBBlackBoxOptimizer(1000, 10)
result = optimizer(func)

# Refine the strategy (continued)
new_individual, new_bounds = result
new_individual, new_bounds = optimizer.refine_strategy(new_individual, result)
print(f"Refined individual: {new_individual}, Refined bounds: {new_bounds}")