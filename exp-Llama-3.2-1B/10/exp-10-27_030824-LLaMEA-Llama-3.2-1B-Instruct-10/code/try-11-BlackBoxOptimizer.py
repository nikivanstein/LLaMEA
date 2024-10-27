# Description: Novel Metaheuristic Algorithm for Black Box Optimization using a novel combination of random walk and linear interpolation
# Code: 
import random
import numpy as np
from scipy.optimize import minimize_scalar
import math

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

    def optimize(self, func, budget, dim):
        # Initialize the population with random points in the search space
        population = [[np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))] for _ in range(100)]

        while len(population) > 0 and budget > 0:
            # Evaluate the fitness of each individual in the population
            fitnesses = [individual[1] for individual in population]

            # Select the fittest individuals to reproduce
            fittest_indices = np.argsort(fitnesses)[-5:]  # Select the top 5 fittest individuals
            fittest_individuals = population[fittest_indices]

            # Create new offspring by linearly interpolating between the fittest individuals
            offspring = []
            for i in range(0, len(fittest_individuals), 2):
                # Select two fittest individuals
                parent1 = fittest_individuals[i]
                parent2 = fittest_individuals[i + 1]

                # Linearly interpolate between the two parents
                child = (1 - i / 2) * parent1 + i / 2 * parent2

                # Add the child to the offspring list
                offspring.append(child)

            # Evaluate the fitness of the new offspring
            new_fitnesses = [individual[1] for individual in offspring]

            # Replace the old population with the new offspring
            population = offspring

            # Update the budget
            budget -= 1

        # Return the fittest individual in the final population
        return population[0][1]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Python code for the BlackBoxOptimizer class:
# ```python
# 
# 
# ```python
# ```python
# ```python
budget = 1000
dim = 5
best_individual = BlackBoxOptimizer(budget, dim).optimize(lambda x: -x[0]**2, budget, dim)
print("Best individual:", best_individual)
print("Best fitness:", -best_individual[0]**2)