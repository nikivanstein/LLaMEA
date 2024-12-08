import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.new_individual = None
        self.best_individual = None
        self.best_fitness = -np.inf
        self.population_size = 100
        self.mutation_rate = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a new individual using the current best individual and the mutation rate
            self.new_individual = self.generate_new_individual()
            # Evaluate the function at the new individual
            func_value = func(self.new_individual)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the new individual is within the budget
            if self.func_evaluations < self.budget:
                # If not, update the best individual if the new individual is better
                if func_value > self.best_fitness:
                    self.best_individual = self.new_individual
                    self.best_fitness = func_value
            # Check if the budget is reached
            if self.func_evaluations >= self.budget:
                # If the budget is reached, return the best individual found so far
                return self.best_individual
        # If the budget is reached, return the best individual found so far
        return self.best_individual

    def generate_new_individual(self):
        # Initialize the new individual with a random point in the search space
        new_individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        # Use the mutation rate to refine the strategy
        if random.random() < self.mutation_rate:
            # Generate a new point in the search space
            new_point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Check if the new point is within the budget
            if new_point[0] < self.search_space[0] or new_point[0] > self.search_space[1] or new_point[1] < self.search_space[0] or new_point[1] > self.search_space[1]:
                # If not, return the new individual
                return new_individual
        return new_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 