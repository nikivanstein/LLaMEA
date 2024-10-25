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
        self.population_history = []

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

            # Update the population history
            self.population_history.append(self.population_size - 1)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy using the population history
        best_individual = max(self.population_history, key=lambda i: self.fitnesses[i])
        return best_individual

    def mutate(self, individual):
        # Mutate the individual using a simple strategy: swap two random elements
        # Refine the strategy using the population history
        best_individual = max(self.population_history, key=lambda i: self.fitnesses[i])
        if random.random() < 0.05:
            i, j = random.sample(range(self.population_size), 2)
            self.population[i], self.population[j] = self.population[j], self.population[i]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Mutation"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# refines the strategy using the population history, and includes mutation to introduce randomness.