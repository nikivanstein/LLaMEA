import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_point=None, iterations=100, mutation_rate=0.01):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Initialize the population with random points in the search space
            population = [initial_point if initial_point is not None else np.random.uniform(self.search_space[0], self.search_space[1]), func(initial_point)]
            # Evaluate the function at each point in the population
            for _ in range(iterations):
                new_population = []
                for individual in population:
                    # Evaluate the function at the point
                    evaluation = func(individual)
                    # Increment the function evaluations
                    self.func_evaluations += 1
                    # Return the point and its evaluation
                    new_population.append((individual, evaluation))
                # Replace the old population with the new one
                population = new_population
            # Return the best individual in the population
            return max(population, key=lambda x: x[1])
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 