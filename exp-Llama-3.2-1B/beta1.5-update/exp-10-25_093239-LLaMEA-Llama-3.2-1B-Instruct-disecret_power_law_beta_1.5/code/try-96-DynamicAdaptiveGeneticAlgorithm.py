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
            new_individual = self.select_next_individual()
            fitness = func(new_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = new_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy using a probability of 0.05405405405405406
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        cumulative_probabilities = [0.1, 0.3, 0.5, 0.9, 1.0]
        r = random.random()
        for i, p in enumerate(probabilities):
            if r < p:
                return self.population[i]

    def mutate(self, individual):
        # Randomly select a mutation point and swap the two alleles
        mutation_point = random.randint(0, self.dim - 1)
        individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Refining Strategy"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.