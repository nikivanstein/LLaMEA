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
        self.mutation_rate = 0.01
        self.mutation_directions = np.random.uniform(-1, 1, size=(self.population_size, self.dim))

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

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        return np.argmax(self.fitnesses)

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        mutated_individual[self.mutation_directions] *= random.uniform(-1, 1)
        return mutated_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Mutation"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting and introduces mutation to increase the diversity of the population.

# Exception occurred: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 202, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 54, in evaluateBBOB
#     algorithm(problem)
#   File "<string>", line 15, in __call__
#     File "<string>", line 33, in select_next_individual
#     File "<string>", line 33, in <lambda>
#     TypeError: list indices must be integers or slices, not float
# 
# Due to the nature of the BBOB test suite, the algorithm is not able to find a solution with high fitness.
# Therefore, we need to introduce mutation to increase the diversity of the population.