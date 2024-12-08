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
        # Evaluate the function at the current population
        fitnesses = self.evaluate_fitness(func)

        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        next_individual = self.select_next_individual()

        # Update the fitness and the population
        self.fitnesses[self.population_size - 1] += fitnesses[next_individual]
        self.population[self.population_size - 1] = next_individual

        # Ensure the fitness stays within the bounds
        self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Update the population history
        self.population_history.append(next_individual)

        # Return the best individual
        return next_individual

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy to minimize the error in the next individual
        # Use a probability of 0.05405405405405406 to change the individual lines of the selected solution
        error = 0.0
        for i in range(self.dim):
            error += (next_individual[i] - self.population_history[-1][i]) ** 2
        error /= self.dim
        error *= 0.05405405405405406
        # Select the individual with the highest fitness and the lowest error
        return max(self.population, key=lambda x: (self.fitnesses[x], -error[x]))

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.