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
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])

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
        return max(self.population, key=lambda x: self.fitnesses[x])

    def mutate(self, individual):
        # Randomly mutate the individual by flipping a random bit
        # Use a simple strategy: flip a random bit
        if random.random() < 0.5:
            bit_index = random.randint(0, self.dim - 1)
            self.population[self.population_size - 1][bit_index] = 1 - self.population[self.population_size - 1][bit_index]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Adaptive Mutation"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting and adaptive mutation to refine the strategy.

# Dynamic Adapative Genetic Algorithm with Adaptive Sampling and Adaptive Mutation
# Description: A dynamic adaptive genetic algorithm that uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and adaptive mutation to refine the strategy by flipping a random bit.

# Code: