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
        self.p = 0.5  # probabilistic selection probability

    def __call__(self, func):
        for _ in range(self.budget):
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            new_individual = self.select_next_individual()

            # Evaluate the function at the new individual
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
        # Use probabilistic selection to refine the strategy
        if random.random() < self.p:
            return max(self.population, key=lambda x: self.fitnesses[x])
        else:
            return np.random.choice(self.population, size=self.dim, replace=False)

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Probabilistic Selection"
# This algorithm uses adaptive sampling and probabilistic selection to optimize black box functions.