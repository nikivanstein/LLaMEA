import random
import math
import numpy as np

class DynamicAdapativeGeneticAlgorithm:
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
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Refine the strategy
        # Calculate the average fitness of the last 10% of individuals
        avg_fitness = np.mean([self.fitnesses[i] for i in range(0, self.population_size, 10) if i < self.population_size - 1])
        # Select the next individual based on the average fitness and the dimension
        next_individual = max(self.population, key=lambda x: (self.fitnesses[x] / avg_fitness, x))

        # Evaluate the function at the next individual
        fitness = func(next_individual)

        # Update the fitness and the population
        self.fitnesses[self.population_size - 1] += fitness
        self.population[self.population_size - 1] = next_individual

        # Ensure the fitness stays within the bounds
        self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

# One-line description: "Dynamic Adapative Genetic Algorithm with Adaptive Sampling and Refining Strategy"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.