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
        self.refining_strategy = self.adaptive_refining_strategy()

    def adaptive_refining_strategy(self):
        # Refining strategy: select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # but also consider the current best individual
        def select_next_individual(current_best):
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # but also consider the current best individual
            return max(self.population, key=lambda x: self.fitnesses[x] if x!= current_best else self.fitnesses[current_best])

        # Initialize the best individual
        best_individual = max(self.population, key=lambda x: self.fitnesses[x])

        # Refine the strategy
        for _ in range(self.budget):
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            # but also consider the current best individual
            next_individual = select_next_individual(best_individual)

            # Evaluate the function at the next individual
            fitness = self.evaluate_fitness(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return best_individual

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        return individual

    def __call__(self, func):
        # Evaluate the function at the best individual
        best_individual = self.refining_strategy()

        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # but also consider the current best individual
        next_individual = max(self.population, key=lambda x: self.fitnesses[x] if x!= best_individual else self.fitnesses[best_individual])

        # Evaluate the function at the next individual
        fitness = func(next_individual)

        # Update the fitness and the population
        self.fitnesses[self.population_size - 1] += fitness
        self.population[self.population_size - 1] = next_individual

        # Ensure the fitness stays within the bounds
        self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return best_individual