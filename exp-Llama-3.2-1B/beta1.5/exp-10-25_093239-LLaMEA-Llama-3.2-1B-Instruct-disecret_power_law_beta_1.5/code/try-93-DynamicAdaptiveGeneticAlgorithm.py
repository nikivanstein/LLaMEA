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

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            new_individual = self.select_next_individual()
            
            # Evaluate the function at the new individual
            fitness = func(new_individual)
            
            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = new_individual
            
            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and then use a more refined strategy to refine its strategy
        # This is done by selecting the individual with the highest fitness
        # that is also within the bounds
        # and then use a greedy approach to refine its strategy
        # by selecting the individual with the highest fitness that is
        # also within the bounds
        return self.select_individual_with_highest_fitness()
        
    def select_individual_with_highest_fitness(self):
        # Select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.