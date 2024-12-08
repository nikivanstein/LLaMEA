import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = np.inf

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
        if random.random() < 0.15:
            # Refine the strategy by changing the fitness of the individual
            self.best_individual = individual
            self.best_fitness = np.inf
            # Generate a new individual with a random fitness
            new_individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the new individual
            new_func_value = func(new_individual)
            # Update the best individual and fitness if necessary
            if new_func_value < self.best_fitness:
                self.best_individual = new_individual
                self.best_fitness = new_func_value
        return individual

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 