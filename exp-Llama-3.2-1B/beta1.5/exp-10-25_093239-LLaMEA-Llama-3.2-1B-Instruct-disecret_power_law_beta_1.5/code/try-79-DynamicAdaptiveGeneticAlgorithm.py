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
            next_individual = self.select_next_individual()

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
        # Refine the strategy by using the adaptive sampling
        # Calculate the adaptive sampling weight
        adaptive_sampling_weight = self.calculate_adaptive_sampling_weight()

        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        next_individual = self.select_next_individual_with_adaptive_sampling(adaptive_sampling_weight)

        # Evaluate the function at the next individual
        fitness = func(next_individual)

        # Update the fitness and the population
        self.fitnesses[self.population_size - 1] += fitness
        self.population[self.population_size - 1] = next_individual

        # Ensure the fitness stays within the bounds
        self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        return next_individual

    def select_next_individual_with_adaptive_sampling(self, adaptive_sampling_weight):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy by using the adaptive sampling
        # Calculate the adaptive sampling weight
        adaptive_sampling_weight = self.calculate_adaptive_sampling_weight()

        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy by using the adaptive sampling
        next_individual = self.select_next_individual_with_adaptive_sampling_weight(adaptive_sampling_weight)

        return next_individual

    def calculate_adaptive_sampling_weight(self):
        # Calculate the adaptive sampling weight
        # Use a simple strategy: select the individual with the highest fitness
        # Calculate the adaptive sampling weight based on the fitness
        return 1 / self.fitnesses[self.population_size - 1]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.