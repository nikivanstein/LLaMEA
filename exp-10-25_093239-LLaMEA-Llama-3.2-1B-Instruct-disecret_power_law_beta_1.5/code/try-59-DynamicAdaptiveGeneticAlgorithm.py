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
            # and then refine it based on the probability of 0.02702702702702703
            probability = 0.02702702702702703
            next_individual = self.select_next_individual(probability)
            fitness = func(next_individual)
            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual
            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self, probability):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

    def mutate(self, individual):
        # Randomly mutate the individual
        # Use a simple strategy: swap two random genes
        index1, index2 = random.sample(range(self.dim), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.