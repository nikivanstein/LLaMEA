import random
import math
import copy

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
        # Use a more refined strategy: select the individual with the highest fitness and the highest probability
        # Use a probability of 0.05405405405405406 to change the individual lines of the selected solution to refine its strategy
        probabilities = [1 / self.fitnesses[i] for i in range(self.population_size)]
        next_individual = self.select_individual_with_probability(probabilities)
        return next_individual

    def select_individual_with_probability(self, probabilities):
        # Select the individual with the highest probability
        next_individual = max(self.population, key=lambda x: probabilities[x])
        return next_individual

# One-line description: "Dynamic Adapative Genetic Algorithm with Adaptive Sampling and Improved Fitness Refinement"
# Code: 