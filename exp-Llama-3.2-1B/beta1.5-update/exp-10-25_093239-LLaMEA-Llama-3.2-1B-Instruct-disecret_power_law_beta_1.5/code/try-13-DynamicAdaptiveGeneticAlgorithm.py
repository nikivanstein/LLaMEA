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
        self.refining_factor = 0.9

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            next_individual = self.select_next_individual()

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Refine the selected individual based on the fitness and the dimension
        refined_individual = self.refine_individual(next_individual)

        # Evaluate the refined individual
        refined_fitness = func(refined_individual)

        # Update the best individual
        self.population[0] = refined_individual
        self.fitnesses[0] += refined_fitness

        # Ensure the fitness stays within the bounds
        self.fitnesses[0] = min(max(self.fitnesses[0], -5.0), 5.0)

        # Return the best individual
        return refined_individual

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        return max(self.population, key=lambda x: self.fitnesses[x])

    def refine_individual(self, individual):
        # Refine the selected individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # and use a probability of 0.9 to refine the strategy
        refined_individual = individual
        for _ in range(self.dim):
            # Select a random individual with a probability of 0.9
            # to refine the strategy
            next_individual = random.choice([i for i in self.population if i!= refined_individual])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            refined_individual = next_individual
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = refined_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        return refined_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Evolutionary Refining"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and uses evolutionary refining to refine the strategy.

# Exception occurred: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 52, in evaluateBBOB
#     algorithm(problem)
#   File "<string>", line 15, in __call__
#     algorithm(problem)
#   File "<string>", line 33, in select_next_individual
#     TypeError: list indices must be integers or slices, not float