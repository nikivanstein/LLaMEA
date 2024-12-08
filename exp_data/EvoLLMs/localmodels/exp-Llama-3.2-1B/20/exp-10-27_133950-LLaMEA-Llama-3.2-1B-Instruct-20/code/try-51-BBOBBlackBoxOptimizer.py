import numpy as np
import random
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.strategies = {
            'random': self.random_strategy,
           'refine': self.refine_strategy,
            'bounded': self.bounded_strategy,
            'gradient_descent': self.gradient_descent_strategy
        }
        self.current_strategy = self.strategies['random']

    def __call__(self, func):
        if self.func_evaluations >= self.budget:
            raise Exception("Optimization limit reached")
        return self.current_strategy(func)

    def random_strategy(self, func):
        x = self.search_space[0]
        for _ in range(self.dim):
            x += random.uniform(-5.0, 5.0)
        return func(x)

    def refine_strategy(self, func):
        x = self.search_space[0]
        for _ in range(self.dim):
            x = x - 0.1 * x
        return func(x)

    def bounded_strategy(self, func):
        x = self.search_space[0]
        for _ in range(self.dim):
            x = max(min(x, 5.0), -5.0)
        return func(x)

    def gradient_descent_strategy(self, func):
        x = self.search_space[0]
        learning_rate = 0.1
        for _ in range(self.dim):
            x = x - learning_rate * func(x)
        return func(x)

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the optimizer with a new strategy
optimizer = BBOBBlackBoxOptimizer(1000, 10)
optimizer.current_strategy = optimizer.refine_strategy
func = lambda x: x**2
result = optimizer(func)
print(result)