import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.new_individuals = []
        self.best_individuals = []

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

    def mutate(self, individual):
        # Refine the strategy by changing 15% of the individual's lines
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < 0.15:
                mutated_individual[i] += random.uniform(-0.1, 0.1)
        return mutated_individual

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        func_value = self.func(individual)
        # Update the best individual if necessary
        if func_value > self.best_individuals[0][0] or func_value < self.best_individuals[1][0]:
            self.best_individuals = [individual, individual]
        return func_value

    def get_best_individual(self):
        # Return the best individual found so far
        return self.best_individuals[0]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np
# import copy
# import time
# from scipy.optimize import differential_evolution

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = (-5.0, 5.0)
#         self.func_evaluations = 0
#         self.new_individuals = []
#         self.best_individuals = []

#     def __call__(self, func):
#         while self.func_evaluations < self.budget:
#             # Generate a random point in the search space
#             point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
#             # Evaluate the function at the point
#             func_value = func(point)
#             # Increment the function evaluations
#             self.func_evaluations += 1
#             # Check if the point is within the budget
#             if self.func_evaluations < self.budget:
#                 # If not, return the point
#                 return point
#         # If the budget is reached, return the best point found so far
#         return self.search_space[0], self.search_space[1]

#     def mutate(self, individual):
#         # Refine the strategy by changing 15% of the individual's lines
#         mutated_individual = copy.deepcopy(individual)
#         for i in range(self.dim):
#             if random.random() < 0.15:
#                 mutated_individual[i] += random.uniform(-0.1, 0.1)
#         return mutated_individual

#     def evaluate_fitness(self, individual):
#         # Evaluate the function at the individual
#         func_value = self.func(individual)
#         # Update the best individual if necessary
#         if func_value > self.best_individuals[0][0] or func_value < self.best_individuals[1][0]:
#             self.best_individuals = [individual, individual]
#         return func_value

#     def get_best_individual(self):
#         # Return the best individual found so far
#         return self.best_individuals[0]

# # Test the algorithm
# optimizer = BlackBoxOptimizer(100, 10)
# func = lambda x: np.sin(x)
# optimizer.__call__(func)
# print(optimizer.get_best_individual())