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
            # and then refine its strategy using a probability of 0.05
            selected_individual = max(self.population, key=lambda x: self.fitnesses[x])
            # Get the current fitness of the selected individual
            current_fitness = func(selected_individual)

            # Refine the strategy using a probability of 0.05
            if random.random() < 0.05:
                # Select a new individual with a lower fitness
                new_individual = random.uniform(-5.0, 5.0)
            else:
                # Select a new individual with a higher fitness
                new_individual = selected_individual

            # Ensure the fitness stays within the bounds
            new_fitness = func(new_individual)
            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += current_fitness
            self.population[self.population_size - 1] = new_individual
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]