import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = [self.search_space] * self.budget
        for _ in range(1):  # Evolve the population for a few generations
            # Evaluate the function at each point in the population
            fitnesses = [func(individual) for individual in population]
            # Select the fittest individuals
            self.func_evaluations += 1
            fittest = population[np.argmax(fitnesses)]
            # Create a new generation by crossover and mutation
            new_population = [fittest[:i] + [random.uniform(self.search_space[0], self.search_space[1])] + fittest[i+1:] for i in range(self.budget)]
            # Replace the old population with the new one
            population = new_population
        # Return the best individual in the population
        return population[np.argmax(fitnesses)]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 