import random
import math
import numpy as np

class AdaptiveAdaptiveGeneticAlgorithm:
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
            # Use a more sophisticated strategy: select the individual with the highest fitness
            #  and the most promising offspring
            offspring = self.select_offspring()

            # Evaluate the function at the offspring
            fitness = func(offspring)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = offspring

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_offspring(self):
        # Select the offspring based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Use a more sophisticated strategy: select the individual with the highest fitness
        #  and the most promising offspring
        offspring = [self.population[i] for i in range(self.population_size) if self.fitnesses[i] == max(self.fitnesses)]

        # Refine the strategy to prioritize promising individuals
        # Use a simple strategy: select the individual with the highest fitness
        # Use a more sophisticated strategy: select the individual with the highest fitness
        #  and the most promising offspring
        offspring = self.refine_offspring(offspring)

        return offspring

    def refine_offspring(self, offspring):
        # Refine the strategy to prioritize promising individuals
        # Use a simple strategy: select the individual with the highest fitness
        # Use a more sophisticated strategy: select the individual with the highest fitness
        #  and the most promising offspring
        promising_individuals = [i for i, _ in enumerate(offspring) if self.fitnesses[i] > 0.5]
        return [self.population[i] for i in random.sample(promising_individuals, 10)]

# One-line description: "Adaptive Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.