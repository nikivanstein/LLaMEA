import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        """Randomly mutate an individual's genes."""
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < 0.05:
                mutated_individual[i] += random.uniform(-1, 1)
        return mutated_individual

    def crossover(self, parent1, parent2):
        """Crossover two parents to create a child."""
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

# Description: "Black Box Optimization using Evolutionary Strategies with Dynamic Search Space and Mutation"
# Code: 
# ```python
# HEBBO(budget, dim)
# def __call__(self, func):
#     while self.func_evaluations < self.budget:
#         self.func_evaluations += 1
#         func_value = func(self.search_space)
#         if np.isnan(func_value) or np.isinf(func_value):
#             raise ValueError("Invalid function value")
#         if func_value < 0 or func_value > 1:
#             raise ValueError("Function value must be between 0 and 1")
#         self.search_space = np.linspace(-5.0, 5.0, self.dim)
#     return func_value

# Initialize HEBBO with a budget of 1000 and a dimension of 10
HEBBO = HEBBO(1000, 10)

# Define a function to be optimized
def func(x):
    return x[0]**2 + x[1]**2

# Evaluate the function 1000 times and print the result
result = HEBBO(1000, 10)(func)
print(result)