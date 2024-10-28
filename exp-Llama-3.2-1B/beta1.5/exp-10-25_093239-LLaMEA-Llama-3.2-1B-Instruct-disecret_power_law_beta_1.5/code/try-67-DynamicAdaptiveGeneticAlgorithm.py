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
            new_individual = max(self.population, key=lambda x: self.fitnesses[x])

            # Evaluate the function at the new individual
            fitness = func(new_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = new_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Append the new individual to the history
            self.population_history.append(new_individual)

            # If the fitness is greater than 0.54, select a new individual with a higher fitness
            if fitness > 0.54:
                # Select the individual with the highest fitness
                new_individual = max(self.population, key=lambda x: self.fitnesses[x])

                # Evaluate the function at the new individual
                fitness = func(new_individual)

                # Update the fitness and the population
                self.fitnesses[self.population_size - 1] += fitness
                self.population[self.population_size - 1] = new_individual

                # Ensure the fitness stays within the bounds
                self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # If the fitness is less than 0.46, select a new individual with a lower fitness
            elif fitness < 0.46:
                # Select the individual with the lowest fitness
                new_individual = min(self.population, key=lambda x: self.fitnesses[x])

                # Evaluate the function at the new individual
                fitness = func(new_individual)

                # Update the fitness and the population
                self.fitnesses[self.population_size - 1] += fitness
                self.population[self.population_size - 1] = new_individual

                # Ensure the fitness stays within the bounds
                self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

# One-line description: "DynamicAdaptiveGeneticAlgorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.