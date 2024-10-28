# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

class IteratedPermutationCooling:
    def __init__(self, budget, dim, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.permutation = None
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random permutation of the search space
            self.permutation = np.random.permutation(self.dim)
            # Evaluate the function at the permutation
            value = func(np.array([self.permutation]))
            # Check if the permutation is within the bounds
            if -5.0 <= value[0] <= 5.0 and -5.0 <= value[1] <= 5.0:
                # If the permutation is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best permutation found so far
        return np.max(func(np.array([self.permutation])))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# Iterated Permutation Cooling Algorithm
# Code: 
# ```python
# ```python
# ```python
# def update_solution(self, new_individual):
#     # Refine the strategy based on the probability of mutation
#     if random.random() < self.mutation_prob:
#         # Perform a mutation on the new individual
#         new_individual[0] = random.uniform(-5.0, 5.0)
#         new_individual[1] = random.uniform(-5.0, 5.0)
#     return new_individual

# ```python
# ```python
# ```python
# def update_solution(self, new_individual):
#     # Refine the strategy based on the probability of mutation
#     if random.random() < 0.45:
#         # Perform a mutation on the new individual
#         new_individual[0] = random.uniform(-5.0, 5.0)
#         new_individual[1] = random.uniform(-5.0, 5.0)
#     return new_individual

# optimizer = IteratedPermutationCooling(budget=100, dim=5)
# optimizer = BlackBoxOptimizer(budget=100, dim=5)

# print(optimizer)