import numpy as np
from scipy.optimize import minimize
import random

class EALSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func, initial_individual, learning_rate=0.1, alpha=0.01, beta=0.1):
        def wrapper(x, learning_rate=0.1, alpha=0.01, beta=0.1):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        new_individual = initial_individual
        for _ in range(self.dim):
            new_individual = self.update_individual(new_individual, wrapper, learning_rate, alpha, beta)
            self.func_evaluations += 1
        return new_individual

    def update_individual(self, individual, wrapper, learning_rate, alpha, beta):
        x = individual
        for _ in range(self.dim):
            if random.random() < alpha:
                x = wrapper(x, learning_rate=learning_rate, alpha=alpha, beta=beta)
            elif random.random() < beta:
                x = wrapper(x, learning_rate=learning_rate, alpha=alpha, beta=beta)
            else:
                x = x
        return x

# Example usage:
optimizer = EALSA(1000, 10)
func = lambda x: x**2
initial_individual = [0.0] * 10
result = optimizer(func, initial_individual)
print(result)