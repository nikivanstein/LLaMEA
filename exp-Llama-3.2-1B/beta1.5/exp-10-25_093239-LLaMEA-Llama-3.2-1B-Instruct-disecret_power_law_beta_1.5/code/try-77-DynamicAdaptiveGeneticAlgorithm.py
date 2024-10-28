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
            # and refine it using the adaptive sampling strategy
            best_individual = self.select_best_individual()
            fitness = func(best_individual)
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = best_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_best_individual(self):
        # Select the individual with the highest fitness
        best_individual = max(self.population, key=lambda x: self.fitnesses[x])

        # Refine the strategy using the adaptive sampling
        # Select the next individual based on the fitness and the dimension
        # and refine it using the adaptive sampling strategy
        next_individual = self.select_next_individual(best_individual)

        return next_individual

    def select_next_individual(self, best_individual):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and refine it using the adaptive sampling strategy
        return max(self.population, key=lambda x: self.fitnesses[x])

# One-line description: "Dynamic Adapative Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.