# Description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
# Code: 
import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def select_new_point(self, func, budget):
        # Select a new point from the search space using a combination of random search and function evaluation
        num_evaluations = min(budget, self.func_evaluations)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations
        new_point = np.random.choice(self.search_space, num_evaluations, replace=False)
        value = func(new_point)
        if value < 1e-10:  # arbitrary threshold
            return new_point
        else:
            return new_point

    def mutate(self, new_point):
        # Randomly mutate a new point in the search space
        mutated_point = new_point + np.random.normal(0, 1, self.dim)
        return mutated_point

    def evolve_population(self, population, budget):
        # Evolve the population using a combination of random search and mutation
        for _ in range(budget):
            # Select a new point from the population using a combination of random search and function evaluation
            new_point = self.select_new_point(func, population[-1].shape[0])
            # Mutate the new point in the search space
            mutated_point = self.mutate(new_point)
            # Add the mutated point to the population
            population.append(mutated_point)