import numpy as np
import random

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
        # Initialize the population with random individuals
        population = [self.generate_individual(func, budget, dim) for _ in range(100)]

        # Evaluate the population for a fixed number of iterations
        for _ in range(10):
            # Select the fittest individual
            fittest_individual = max(population, key=lambda individual: individual[1])

            # Perform a random walk
            new_individual = fittest_individual
            for _ in range(10):
                # Generate a random point in the search space
                point = np.random.uniform(self.search_space[0], self.search_space[1])
                # Evaluate the function at the point
                evaluation = func(point)
                # Update the new individual
                new_individual = (1 - 0.1 * (new_individual[0] + 0.1 * (point - new_individual[0])) + 0.1 * (point - fittest_individual[0]), evaluation)

            # Add the new individual to the population
            population.append(new_individual)

            # Update the fittest individual
            fittest_individual = max(population, key=lambda individual: individual[1])

        # Return the fittest individual
        return fittest_individual

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: