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
        # Adaptive sampling: select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and refine the strategy by using a probabilistic approach
        prob = 0.1  # probability of refining the strategy
        refines = 0  # number of times the strategy is refined

        while refines < self.budget:
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Refine the strategy
            if random.random() < prob:
                # Select the next individual based on the fitness and the dimension
                # Use a probabilistic approach to refine the strategy
                refines += 1
                next_individual = max(self.population, key=lambda x: self.fitnesses[x])
                fitness = func(next_individual)
                self.fitnesses[self.population_size - 1] += fitness
                self.population[self.population_size - 1] = next_individual

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

# One-line description: "DynamicAdaptiveGeneticAlgorithm with Adaptive Sampling and Refining Strategy"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.