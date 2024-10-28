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

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # and then use a more refined strategy to select the next individual
            # based on the fitness and the dimension
            best_individual = max(self.population, key=lambda x: self.fitnesses[x])
            fitness = func(best_individual)
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = best_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Refine the strategy
            # Select the next individual based on the fitness and the dimension
            # Use a weighted sum of the fitness and the dimension
            # with weights that decrease over time
            weights = [1.0] * self.dim
            weights[-1] *= 0.9
            weights[:-1] *= 0.1
            best_individual = max(self.population, key=lambda x: (x, weights[0] * self.fitnesses[x], weights[-1] * self.fitnesses[x + 1]))
            self.fitnesses[self.population_size - 1] += weights[-1] * self.fitnesses[x + 1]
            self.population[self.population_size - 1] = best_individual

        # Return the best individual
        return self.population[0]

# One-line description: "Dynamic Adapative Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.