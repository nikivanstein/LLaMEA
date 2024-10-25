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
        # Initialize the population with random values
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]

        for _ in range(self.budget):
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # If the fitness is below the lower bound, select the individual with the lowest fitness
            # If the fitness is above the upper bound, select the individual with the highest fitness
            # Use a probability of 0.05405405405405406 to refine the strategy
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
        # If the fitness is below the lower bound, select the individual with the lowest fitness
        # If the fitness is above the upper bound, select the individual with the highest fitness
        # Use a probability of 0.05405405405405406 to refine the strategy
        # Use a bias towards the lower bound and a bias towards the upper bound
        bias = 0.3
        if self.fitnesses[self.population_size - 1] < -5.0:
            bias = 0.2
        elif self.fitnesses[self.population_size - 1] > 5.0:
            bias = 0.2
        next_individual = max(self.population, key=lambda x: self.fitnesses[x])

        # Refine the strategy
        if random.random() < 0.05405405405405406:
            bias = 0.1
        if random.random() < 0.05405405405405406:
            bias = -0.1

        return bias * next_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.