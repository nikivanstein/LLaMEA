import random
import math
import numpy as np

class AdaptiveBlackBoxOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size
        self.population_history = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize the population history
            self.population_history.append(self.population[:])

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

            # Update the population history
            self.population_history[-1].append(next_individual)

            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = max(self.population_history, key=lambda x: x[-1])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

# One-line description: "Adaptive Black Box Optimization using Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.