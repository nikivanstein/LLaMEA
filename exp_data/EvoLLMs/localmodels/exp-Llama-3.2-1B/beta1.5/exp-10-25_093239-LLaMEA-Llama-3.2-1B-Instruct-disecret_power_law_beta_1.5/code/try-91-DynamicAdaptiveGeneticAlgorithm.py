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
        # Initialize the population with random function evaluations
        self.population = [func(x) for x in random.uniform(-5.0, 5.0) for _ in range(self.population_size)]

        # Adaptive sampling: select the next individual based on the fitness and the dimension
        for _ in range(self.budget):
            # Calculate the average fitness
            avg_fitness = sum(self.fitnesses) / len(self.fitnesses)

            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # If the average fitness is below the threshold, select the individual with the highest fitness
            next_individual = max(self.population, key=lambda x: self.fitnesses[x] if self.fitnesses[x] / avg_fitness > 0.5 else self.fitnesses[x])

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += max(0, self.fitnesses[self.population_size - 1] - avg_fitness)
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.