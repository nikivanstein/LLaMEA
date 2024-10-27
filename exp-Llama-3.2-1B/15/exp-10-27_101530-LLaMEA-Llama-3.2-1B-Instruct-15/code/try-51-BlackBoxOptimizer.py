import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_point = None
        self.best_fitness = -np.inf

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
            # Check if the point is better than the best point found so far
            if func_value > self.best_fitness:
                self.best_point = point
                self.best_fitness = func_value
        # If the budget is reached, return the best point found so far
        return self.best_point

def evaluateBBOB(func, budget, dim, algorithm):
    # Initialize the algorithm
    algorithm = BlackBoxOptimizer(budget, dim)
    # Evaluate the function for the algorithm
    individual = algorithm(func)
    # Return the fitness of the individual
    return algorithm.__call__(func)(individual)

def mutation_exp(func, budget, dim, algorithm, mutation_rate):
    # Initialize the algorithm
    algorithm = BlackBoxOptimizer(budget, dim)
    # Evaluate the function for the algorithm
    individual = algorithm(func)
    # Return the fitness of the individual
    return algorithm.__call__(func)(individual)

# Define a novel heuristic algorithm
class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_point = None
        self.best_fitness = -np.inf

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
            # Check if the point is better than the best point found so far
            if func_value > self.best_fitness:
                self.best_point = point
                self.best_fitness = func_value
        # If the budget is reached, return the best point found so far
        return self.best_point

# Update the selected solution
NovelMetaheuristicOptimizer(100, 5, 0.1).__call__(lambda x: x**2)

# One-line description
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the strategy by incorporating mutation and probability-based changes