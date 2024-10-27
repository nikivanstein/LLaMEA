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
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def adapt_strategy(self, individual):
        # Refine the strategy by changing individual lines of the selected solution
        # to refine its strategy
        if random.random() < 0.2:
            # Exploration strategy
            new_individual = individual.copy()
            new_individual[0] = random.uniform(-5.0, 5.0)
            return new_individual
        else:
            # Exploitation strategy
            return individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the optimizer with a new solution
new_individual = optimizer.adapt_strategy(result)
optimizer = BBOBBlackBoxOptimizer(1000, 10)
result = optimizer.func(result)
print(result)