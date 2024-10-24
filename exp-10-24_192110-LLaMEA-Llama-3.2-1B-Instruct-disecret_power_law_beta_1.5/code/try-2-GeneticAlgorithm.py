import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_probability = 0.05
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        # Initialize the population with random solutions
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the black box function for each solution in the population
        scores = [func(solution) for solution in self.population]
        # Select the top-scoring solutions
        top_solutions = sorted(zip(self.population, scores), key=lambda x: x[1], reverse=True)[:self.budget]
        # Create a new population by mutating the top solutions
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_solutions, 2)
            child = self.mutate(parent1, parent2, self.mutation_probability)
            new_population.append(child)
        return new_population

    def mutate(self, solution, parent1, parent2):
        # Select a random individual from the parent solutions
        parent1, parent2 = random.sample([solution, solution], 2)
        # Generate a new child solution by combining the two parent solutions
        child = [x + y for x, y in zip(parent1, parent2)]
        # Ensure the child solution stays within the search space
        child = [max(-5.0, min(5.0, x)) for x in child]
        return child

# Description: Genetic Algorithm using Evolutionary Optimization
# Code: 