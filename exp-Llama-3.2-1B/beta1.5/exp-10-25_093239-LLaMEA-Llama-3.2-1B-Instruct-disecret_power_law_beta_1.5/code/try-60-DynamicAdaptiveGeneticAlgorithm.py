import random
import math

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
            next_individual = self.select_next_individual()

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and refine the strategy based on the fitness values
        best_individual = self.population[0]
        best_fitness = self.fitnesses[0]
        for i, fitness in enumerate(self.fitnesses):
            if fitness > best_fitness:
                best_individual = self.population[i]
                best_fitness = fitness

        # Refine the strategy based on the fitness values
        # Use a simple strategy: select the individual with the highest fitness
        # and select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and refine the strategy based on the fitness and the dimension
        next_individual = self.select_next_individual()

        # Evaluate the function at the next individual
        fitness = func(next_individual)

        # Update the fitness and the population
        self.fitnesses[self.population_size - 1] += fitness
        self.population[self.population_size - 1] = next_individual

        # Ensure the fitness stays within the bounds
        self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return best_individual

# One-line description: "Dynamic Adapative Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.