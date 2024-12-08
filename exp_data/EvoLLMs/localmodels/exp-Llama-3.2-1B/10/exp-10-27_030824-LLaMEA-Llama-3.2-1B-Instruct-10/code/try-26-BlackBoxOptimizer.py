import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations=1000, step_size=0.01, exploration_rate=0.1):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Evaluate the function for each individual in the population
        for _ in range(iterations):
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1

            # Select the fittest individual based on the evaluation
            fittest_point = population[np.argmin([evaluation for point, evaluation in zip(population, func)])]
            # Create a new individual by linearly interpolating between the fittest point and the current point
            new_point = fittest_point + (point - fittest_point) * exploration_rate
            # Update the population with the new individual
            population.append(new_point)

        # Select the fittest individual based on the evaluation
        fittest_point = population[np.argmin([evaluation for point, evaluation in zip(population, func)])]
        # Return the fittest individual and its evaluation
        return fittest_point, func(fittest_point)

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation with adaptive step size and exploration rate.

# Code: