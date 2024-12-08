import random
import math

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size
        self.refined_individuals = []
        self.refined_fitnesses = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])
            # Refine the strategy by selecting the next individual based on the fitness and the dimension,
            # and ensure the fitness stays within the bounds
            refined_individual = self.refine_strategy(next_individual)
            # Evaluate the function at the refined individual
            fitness = func(refined_individual)
            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = refined_individual
            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)
            # Store the refined individual and its fitness
            self.refined_individuals.append(refined_individual)
            self.refined_fitnesses.append(fitness)

        # Return the best individual
        return self.refined_individuals[0]

    def refine_strategy(self, individual):
        # Use a simple strategy: select the individual with the highest fitness and a random offset
        # between -1.0 and 1.0
        offset = random.uniform(-1.0, 1.0)
        return individual + offset

# One-line description: "Adaptive Genetic Algorithm with Refining Strategy"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and also refines its strategy
# to improve its performance.