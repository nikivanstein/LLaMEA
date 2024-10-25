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
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy by using a weighted average of the fitness and the dimension
        weights = [0.5, 0.3, 0.2]  # weights for fitness, dimension, and noiseless function
        return [self.fitnesses[i] / weights[i] for i in range(self.population_size)]

    def mutate(self, individual):
        # Mutate the individual by adding a small random noise
        return individual + random.uniform(-0.1, 0.1)

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        return func(individual)

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Refining Strategy"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, while refining its strategy
# by using a weighted average of the fitness and the dimension.