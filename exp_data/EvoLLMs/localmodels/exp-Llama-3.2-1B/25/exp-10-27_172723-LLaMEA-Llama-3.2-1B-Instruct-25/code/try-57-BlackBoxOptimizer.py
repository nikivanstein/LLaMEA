import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def adaptive_search(self, func, initial_point, budget):
        # Initialize the population with random points
        population = [initial_point + np.random.uniform(-1, 1, self.dim) for _ in range(100)]

        # Evaluate the function for each point in the population
        for _ in range(budget):
            min_value = np.inf
            best_point = None
            for population_point in population:
                value = func(population_point)
                if value < min_value:
                    min_value = value
                    best_point = population_point

            # Refine the search space based on the minimum value
            new_search_space = np.linspace(min_value - 1, min_value + 1, self.dim)
            new_population = [population_point + np.random.uniform(-1, 1, self.dim) for population_point in population]
            for new_point in new_population:
                value = func(new_point)
                if value < min_value:
                    min_value = value
                    best_point = new_point

            # Replace the worst point in the population with the new point
            population[population.index(best_point)] = new_point

        # Return the best point in the population
        return population[0]

# One-line description: "Metaheuristic Optimization using Adaptive Search and Function Evaluation"