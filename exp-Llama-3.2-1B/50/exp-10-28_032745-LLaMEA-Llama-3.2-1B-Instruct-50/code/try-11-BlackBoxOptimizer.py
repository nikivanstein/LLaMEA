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
        # Initialize population with random individuals
        population = [copy.deepcopy(initial_guess) for _ in range(50)]

        # Evaluate initial population
        for _ in range(iterations):
            if _ >= self.budget:
                break
            fitness_values = [func(individual, self.func) for individual in population]
            population = [individual for _, individual in zip(fitness_values, population) if individual[fitness_values.index(max(fitness_values))] < self.func(population[0], self.func)]

        # Refine the population using the 0.45 probability
        refined_population = []
        for _ in range(iterations):
            if _ >= self.budget:
                break
            new_individual = copy.deepcopy(population[0])
            if random.random() < 0.45:
                # Select the best individual in the population
                best_individual = max(population, key=lambda individual: individual[fitness_values.index(max(fitness_values))])
                # Select two individuals from the search space
                idx1, idx2 = random.sample(range(self.dim), 2)
                # Perform a swap operation
                new_individual[idx1], new_individual[idx2] = best_individual[idx1], best_individual[idx2]
            # Evaluate the new individual
            fitness_value = self.func(new_individual, self.func)
            # Add the new individual to the refined population
            refined_population.append(new_individual)
            # Update the population
            population = refined_population

        # Return the best individual in the refined population
        return min(refined_population, key=lambda individual: individual[fitness_values.index(max(fitness_values))])

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# The algorithm uses a combination of random swaps and elitism to refine the population, with a 0.45 probability of selecting the best individual in the population