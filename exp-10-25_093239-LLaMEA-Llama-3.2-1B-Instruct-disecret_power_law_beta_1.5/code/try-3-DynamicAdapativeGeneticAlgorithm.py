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
        self.refining_dim = 0

    def __call__(self, func):
        for _ in range(self.budget):
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

            # Refine the strategy based on the fitness
            if self.fitnesses[self.population_size - 1] / self.budget < 0.054:
                self.refining_dim = 1
            elif self.fitnesses[self.population_size - 1] / self.budget > 0.054:
                self.refining_dim = -1

            # Update the population with the refining strategy
            if self.refining_dim == 1:
                self.population = [self.refine_individual(next_individual, self.dim) for _ in range(self.population_size)]
            elif self.refining_dim == -1:
                self.population = [self.refine_individual(next_individual, self.dim + 1) for _ in range(self.population_size)]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

    def refine_individual(self, individual, dim):
        # Refine the individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness in the current dimension
        return [individual]