import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def novel_metaheuristic(self, func, budget, dim):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Evaluate the function for each individual in the population
        for individual in population:
            evaluation = func(individual)
            # If the evaluation exceeds the budget, return a default point
            if evaluation > budget:
                return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))
            # Update the individual with the evaluation
            population[population.index(individual)] = individual, evaluation

        # Select the fittest individuals
        fittest_individuals = sorted(population, key=lambda x: x[1], reverse=True)[:self.budget]

        # Create a new population by applying the search space transformation to the fittest individuals
        new_population = [individual[0] + random.uniform(-1, 1) for individual in fittest_individuals]

        # Evaluate the function for each individual in the new population
        for individual in new_population:
            evaluation = func(individual)
            # If the evaluation exceeds the budget, return a default point
            if evaluation > budget:
                return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))
            # Update the individual with the evaluation
            new_population[new_population.index(individual)] = individual, evaluation

        # Return the new population
        return new_population

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Code: