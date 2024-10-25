import random
import math
import numpy as np

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func, bounds):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # and refine it by moving towards the optimal solution
            next_individual = self.select_next_individual(bounds)
            # Evaluate the function at the next individual
            fitness = func(next_individual)
            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self, bounds):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and refine it by moving towards the optimal solution
        # Use a weighted sum of the fitness and the bounds
        weights = np.array([1.0 / self.dim] * self.dim)
        weights /= np.sum(weights)
        next_individual = np.random.choice(self.population_size, p=weights)
        # Refine the individual by moving towards the optimal solution
        bounds_refined = np.array([
            (bounds[0] + 0.1 * (bounds[1] - bounds[0])),  # lower bound
            (bounds[0] + 0.1 * (bounds[1] - bounds[0])),  # upper bound
        ])
        next_individual = np.array([bounds_refined[0], bounds_refined[1]]) + 0.01 * (bounds[1] - bounds[0])
        return next_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.