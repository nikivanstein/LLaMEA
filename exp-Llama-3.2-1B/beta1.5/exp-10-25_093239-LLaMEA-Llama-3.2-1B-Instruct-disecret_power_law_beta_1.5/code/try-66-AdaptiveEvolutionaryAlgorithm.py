import random
import math

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func):
        # Initialize the population with random individuals
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]

        # Initialize the best individual and its fitness
        best_individual = self.population[0]
        best_fitness = self.fitnesses[0]

        # Iterate until the budget is exhausted
        for _ in range(self.budget):
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

            # Update the best individual and its fitness
            if fitness > best_fitness:
                best_individual = next_individual
                best_fitness = fitness

        # Return the best individual
        return best_individual

# One-line description: "Adaptive Evolutionary Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.