import random
import math
import numpy as np

class GeneticAlgorithm:
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
        # and refine it using mutation
        best_individual = max(self.population, key=lambda x: self.fitnesses[x])
        best_individual_fitness = self.fitnesses[x]
        # Calculate the mutation rate
        mutation_rate = 0.01
        # Refine the individual using mutation
        for _ in range(10):
            # Select a random individual
            new_individual = random.uniform(-5.0, 5.0)
            # Calculate the fitness of the new individual
            fitness = func(new_individual)
            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = new_individual
            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)
            # Update the best individual
            best_individual = max(self.population, key=lambda x: self.fitnesses[x])

        # Return the best individual with the refined strategy
        return best_individual, best_individual_fitness, mutation_rate

    def mutate(self, individual):
        # Mutate the individual with a probability of mutation_rate
        if random.random() < self.mutation_rate:
            # Select a random mutation point
            mutation_point = random.randint(0, self.dim)
            # Mutate the individual
            individual[mutation_point] = random.uniform(-5.0, 5.0)
        return individual

# One-line description: "Genetic Algorithm with Adaptive Fitness Selection and Mutation"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and uses mutation to refine the strategy.