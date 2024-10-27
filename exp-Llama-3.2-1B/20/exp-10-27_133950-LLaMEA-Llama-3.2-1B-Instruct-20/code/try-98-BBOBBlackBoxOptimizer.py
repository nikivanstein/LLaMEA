import numpy as np
from scipy.optimize import minimize
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.search_space_copy = self.search_space.copy()

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

    def mutate(self, individual):
        if random.random() < 0.2:
            self.search_space_copy = self.search_space.copy()
            for i in range(self.dim):
                self.search_space_copy[i] += random.uniform(-0.1, 0.1)
            individual = individual + self.search_space_copy
        return individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the solution with a mutation strategy
optimizer = BBOBBlackBoxOptimizer(1000, 10)
for _ in range(100):
    result = optimizer(func)
    print(result)