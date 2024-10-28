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
            # and use a probability of 0.08108108108108109 to refine the strategy
            probability = 0.08108108108108109
            new_individual = max(self.population, key=lambda x: self.fitnesses[x], 
                                key=lambda x: self.fitnesses[x] / (x - self.population[0]))
            self.population = [new_individual] * (self.population_size - 1) + [new_individual]
            self.fitnesses = [self.fitnesses[0]] + [self.fitnesses[x] / (x - self.population[0]) for x in self.population[1:]]
            # Ensure the fitness stays within the bounds
            self.fitnesses[0] = min(max(self.fitnesses[0], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.