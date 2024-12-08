import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def novel_metaheuristic_algorithm(self, func, budget):
        # Initialize the population with random points in the search space
        population = [self.search_space[0]] * self.dim + [self.search_space[1]] * self.dim
        for _ in range(self.budget // 2):  # Start with half the budget
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest_individuals = [population[i] for i, fitness in enumerate(fitness) if fitness == max(fitness)]
            # Create a new population by combining the fittest individuals with new individuals
            new_population = []
            for _ in range(self.dim):
                # Randomly select a parent from the fittest individuals
                parent1 = random.choice(fittest_individuals)
                parent2 = random.choice(fittest_individuals)
                # Create a new individual by combining the two parents
                new_individual = tuple((a + b) / 2 for a, b in zip(parent1, parent2))
                # Add the new individual to the new population
                new_population.append(new_individual)
            # Replace the old population with the new one
            population = new_population
        # Return the fittest individual in the final population
        return population[0]