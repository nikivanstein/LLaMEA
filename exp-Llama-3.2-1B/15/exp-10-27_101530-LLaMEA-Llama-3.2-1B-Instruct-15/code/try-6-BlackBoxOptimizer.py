import random
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_individual = None

    def __call__(self, func, initial_individual=None):
        if initial_individual is None:
            initial_individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (initial_individual[0] + random.uniform(-1.0, 1.0), initial_individual[1] + random.uniform(-1.0, 1.0))
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
        if self.best_individual is None:
            self.best_individual = individual
        else:
            best_point = self.best_individual
            if individual < best_point:
                best_point = individual
        mutated_individual = (individual[0] + random.uniform(-1.0, 1.0), individual[1] + random.uniform(-1.0, 1.0))
        return mutated_individual

    def evaluate_fitness(self, func, initial_individual=None):
        if initial_individual is None:
            initial_individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        return func(initial_individual)

    def update_best_individual(self, func, initial_individual=None):
        if initial_individual is None:
            initial_individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        best_individual = func(initial_individual)
        if best_individual < self.search_space[0]:
            self.best_individual = (self.search_space[0], self.search_space[1])
        return best_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 