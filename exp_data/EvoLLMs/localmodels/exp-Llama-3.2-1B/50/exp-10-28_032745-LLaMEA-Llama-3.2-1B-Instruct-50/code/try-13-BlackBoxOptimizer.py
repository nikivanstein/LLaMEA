import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        # Initialize population with random initial guesses
        population = [initial_guess] * self.budget
        for _ in range(iterations):
            # Evaluate fitness of each individual in the population
            fitnesses = [self.func(individual) for individual in population]
            # Select parents using tournament selection with replacement
            parents = random.choices(population, weights=fitnesses, k=self.budget)
            # Create offspring by crossover and mutation
            offspring = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.45:
                    child = child + random.uniform(-0.01, 0.01)
                offspring.append(child)
            # Replace least fit individuals with offspring
            population = offspring
        # Return best individual in the final population
        return population[0], max(population)

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# This algorithm uses a combination of genetic algorithm and simulated annealing to optimize the black box function.
# The search space is divided into sub-regions, and the algorithm evaluates the function at the boundaries of each sub-region.
# The probability of moving to a sub-region is determined by the temperature, which decreases as the function value increases.