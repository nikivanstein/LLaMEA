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
        self.fitness_history = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = self.select_next_individual()

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Store the fitness history
            self.fitness_history.append(self.fitnesses[self.population_size - 1])

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy to refine its strategy
        # Use a probability of 0.08108108108108109 to change the individual lines of the selected solution
        probability = 0.08108108108108109
        if random.random() < probability:
            # Select the individual with the highest fitness
            return max(self.population, key=lambda x: self.fitnesses[x])
        else:
            # Select the individual with the second highest fitness
            return min(self.population, key=lambda x: self.fitnesses[x])

    def mutate(self, individual):
        # Mutate the individual with a probability of 0.1
        if random.random() < 0.1:
            # Change the individual's fitness by 10%
            self.fitnesses[self.population_size - 1] += random.uniform(-0.1, 0.1)
            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

# One-line description: "DynamicAdaptiveGeneticAlgorithm with Adaptive Sampling and Fitness Refining"
# Code: 