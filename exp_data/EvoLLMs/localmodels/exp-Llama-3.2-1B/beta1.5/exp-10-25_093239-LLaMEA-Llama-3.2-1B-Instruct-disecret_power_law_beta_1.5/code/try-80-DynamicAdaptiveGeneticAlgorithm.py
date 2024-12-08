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
            # Refine the strategy based on the fitness history
            fitness = func(self.population[0])
            self.fitness_history.append(fitness)
            updated_individual = self.select_next_individual(fitness)
            self.population[0] = updated_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[0] = min(max(self.fitnesses[0], -5.0), 5.0)

            # Update the fitness history
            self.fitness_history = np.array(self.fitness_history)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self, fitness):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy based on the fitness history
        # Use a more sophisticated strategy, such as:
        # - Select the individual with the highest fitness
        # - Select the individual with the lowest fitness
        # - Select the individual with the highest fitness in the previous generation
        # - Select the individual with the lowest fitness in the previous generation
        # - Select the individual with the highest fitness in the current generation
        # - Select the individual with the lowest fitness in the current generation
        if len(self.fitness_history) > 1:
            best_individual = self.select_best_individual(self.fitness_history)
        else:
            best_individual = self.select_best_individual(self.fitness_history[0])
        return best_individual

    def select_best_individual(self, fitness_history):
        # Select the best individual based on the fitness history
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy based on the fitness history
        # Use a more sophisticated strategy, such as:
        # - Select the individual with the highest fitness
        # - Select the individual with the lowest fitness
        # - Select the individual with the highest fitness in the previous generation
        # - Select the individual with the lowest fitness in the previous generation
        # - Select the individual with the highest fitness in the current generation
        # - Select the individual with the lowest fitness in the current generation
        # - Select the individual with the highest fitness in the current generation
        # - Select the individual with the lowest fitness in the current generation
        best_individual = fitness_history.index(max(fitness_history))
        return best_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.