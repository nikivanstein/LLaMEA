import numpy as np
from scipy.optimize import minimize
from collections import deque

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

    def refine_strategy(self, new_individual):
        # Change the individual line of the selected solution to refine its strategy
        new_individual[0] = np.clip(new_individual[0], self.search_space[0] - 0.1, self.search_space[0] + 0.1)

        # Update the population with the new individual
        self.search_space = np.linspace(self.search_space[0] - 0.1, self.search_space[0] + 0.1, 100)
        self.func_evaluations = 0
        return self

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
optimizer = optimizer(func)

# Refine the solution 10 times
for _ in range(10):
    optimizer.refine_strategy(optimizer.func)

# Evaluate the final solution
result = optimizer(func)
print(result)

# Print the updated population
print("Updated Population:")
for individual in optimizer.search_space:
    print(individual)