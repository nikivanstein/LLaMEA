import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.iterations = 0
        self.cooling_rate = 0.05

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

    def iterated_permutation(self):
        # Refine the strategy using iterated permutation
        while self.iterations < 100:
            # Generate a random permutation of the current population
            permutation = list(range(self.dim))
            random.shuffle(permutation)
            # Evaluate the new population
            new_individuals = [self.evaluateBBOB(func, [point] for point in permutation)]
            # Update the current population
            self.current_population = new_individuals
            self.iterations += 1
            # Refine the strategy based on the fitness of the new population
            if self.best_fitness < np.max([np.max(func(new_individual)) for new_individual in new_individuals]):
                self.best_individual = permutation
                self.best_fitness = np.max([np.max(func(new_individual)) for new_individual in new_individuals])
            # Cool down the strategy
            self.cooling_rate *= 0.95
        return self.best_individual

    def evaluateBBOB(self, func, individuals):
        # Evaluate the black box function at each individual in the population
        fitnesses = [func(individual) for individual in individuals]
        # Return the fitness of the best individual
        return np.max(fitnesses)

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 