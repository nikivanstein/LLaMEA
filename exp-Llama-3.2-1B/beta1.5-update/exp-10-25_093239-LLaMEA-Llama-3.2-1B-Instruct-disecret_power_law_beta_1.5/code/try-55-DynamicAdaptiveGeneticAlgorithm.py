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

    def __call__(self, func, max_iter=1000):
        for _ in range(max_iter):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # and then use a weighted average of the fitness with the current fitness
            # to refine the strategy
            new_individual = self.select_next_individual()
            fitness = func(new_individual)
            weights = [0.5, 0.5]  # weights for adaptive sampling
            updated_fitness = fitness * weights[0] + (1 - weights[0]) * fitness
            self.fitnesses[self.population_size - 1] += updated_fitness
            self.population[self.population_size - 1] = new_individual
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and then use a weighted average of the fitness with the current fitness
        # to refine the strategy
        return max(self.population, key=lambda x: self.fitnesses[x])

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.