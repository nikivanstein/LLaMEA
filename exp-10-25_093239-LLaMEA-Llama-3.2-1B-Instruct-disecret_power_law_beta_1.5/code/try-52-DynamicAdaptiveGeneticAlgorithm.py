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
        self.tuning_threshold = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # and use the tuning threshold to adjust the strategy
            best_individual = max(self.population, key=lambda x: self.fitnesses[x])
            fitness = func(best_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = best_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Tuning the strategy: adjust the fitness function based on the fitness and the dimension
            if fitness / self.fitnesses[self.population_size - 1] > self.tuning_threshold:
                # Increase the dimension
                self.dim += 1
                self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
                self.fitnesses = [0] * self.population_size
            else:
                # Decrease the dimension
                self.dim -= 1
                self.population = [best_individual for _ in range(self.population_size)]
                self.fitnesses = [0] * self.population_size

        # Return the best individual
        return self.population[0]

# One-line description: "Dynamic Adapative Genetic Algorithm with Adaptive Sampling and Tuning"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and adjusts the strategy based on the fitness and the dimension to improve the search space.