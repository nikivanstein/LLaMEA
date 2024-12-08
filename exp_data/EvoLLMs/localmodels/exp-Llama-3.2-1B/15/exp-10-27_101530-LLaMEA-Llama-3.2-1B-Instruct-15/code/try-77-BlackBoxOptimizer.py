import random
import numpy as np
from collections import deque
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.search_space_history = deque(maxlen=self.budget)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, new_individual):
        # Refine the strategy by changing the first and last elements of the individual
        new_individual = new_individual[:self.dim//2] + [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim//2)] + new_individual[-self.dim//2:]
        return new_individual

    def __next__(self):
        while True:
            new_individual = self.evaluate_fitness(self.search_space_history)
            if new_individual not in self.search_space_history:
                self.search_space_history.append(new_individual)
                return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        func_value = func(individual)
        return func_value

def func(individual):
    # Generate a random noiseless function
    return np.sin(individual[0]) * np.cos(individual[1])

def minimize_bbo(func, bounds, initial_point, budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    for _ in range(budget):
        point = optimizer.__next__()
        if func(point) > func(initial_point):
            initial_point = point
    return initial_point

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize
# import math

# def func(individual):
#     # Generate a random noiseless function
#     return np.sin(individual[0]) * np.cos(individual[1])

# def minimize_bbo(func, bounds, initial_point, budget, dim):
#     optimizer = BlackBoxOptimizer(budget, dim)
#     for _ in range(budget):
#         point = optimizer.__next__()
#         if func(point) > func(initial_point):
#             initial_point = point
#     return initial_point

# def mutate_bbo(individual):
#     # Refine the strategy by changing the first and last elements of the individual
#     new_individual = individual[:individual.shape[0]//2] + [random.uniform(bounds[0][0], bounds[0][1]) for _ in range(individual.shape[0]//2)] + individual[-individual.shape[0]//2:]
#     return new_individual

# def main():
#     # Set the bounds and budget
#     bounds = ((-5.0, 5.0), (-5.0, 5.0))
#     budget = 100
#     dim = 2
#     initial_point = (-3.0, -4.0)
#     best_individual = None
#     best_fitness = float('inf')
#     for _ in range(1000):
#         best_individual = minimize_bbo(func, bounds, initial_point, budget, dim)
#         best_fitness = func(best_individual)
#         if best_fitness < best_fitness:
#             initial_point = best_individual
#     print(f"Best individual: {best_individual}")
#     print(f"Best fitness: {best_fitness}")

# main()