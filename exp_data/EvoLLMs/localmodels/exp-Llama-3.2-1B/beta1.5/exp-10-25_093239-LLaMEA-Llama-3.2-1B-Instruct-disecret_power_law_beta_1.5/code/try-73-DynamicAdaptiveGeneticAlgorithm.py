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
        self.tolerance = 0.1  # tolerance for convergence

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            next_individual = self.select_next_individual()

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Update the population using the adaptive sampling strategy
            if random.random() < 0.1:  # 10% chance to select the next individual based on the fitness
                next_individual = self.select_next_individual()

            # Update the best individual
            self.population[0] = next_individual

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and use a probabilistic strategy to refine the strategy
        fitnesses = [self.fitnesses[i] for i in range(self.population_size)]
        probabilities = [fitness / fitnesses[i] for i in range(self.population_size)]
        probabilities = [p if i == 0 else p * probabilities[i - 1] for i, p in enumerate(probabilities)]
        r = random.random()
        cumulative_probability = 0
        for i, p in enumerate(probabilities):
            cumulative_probability += p
            if r <= cumulative_probability:
                return self.population[i]
        return self.population[-1]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.