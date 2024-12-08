import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        # Define the mutation strategy based on the budget and dimensionality
        if self.budget < 10:
            mutation_rate = 0.1
        elif self.budget < 50:
            mutation_rate = 0.05
        else:
            mutation_rate = 0.01

        # Initialize the population with random individuals
        population = [initial_guess.copy() for _ in range(50)]

        for _ in range(iterations):
            # Evaluate the fitness of each individual
            fitness = [self.func(individual) for individual in population]

            # Select the fittest individuals for mutation
            fittest_individuals = population[np.argsort(fitness)[:self.budget]]

            # Perform mutation on the fittest individuals
            mutated_individuals = []
            for individual in fittest_individuals:
                for _ in range(mutation_rate * self.dim):
                    new_individual = copy.deepcopy(individual)
                    if random.random() < 0.5:  # Refine the strategy by changing the line
                        new_individual[0] += random.uniform(-0.01, 0.01)
                    mutated_individuals.append(new_individual)

            # Replace the least fit individuals with the mutated ones
            population = mutated_individuals

        # Return the best individual in the final population
        return population[np.argmin(fitness)]

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 