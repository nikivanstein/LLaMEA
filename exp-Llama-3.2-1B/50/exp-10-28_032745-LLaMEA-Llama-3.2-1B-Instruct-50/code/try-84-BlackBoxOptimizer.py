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
        population_size = 100
        population = [initial_guess] * population_size
        fitnesses = []
        for _ in range(iterations):
            for ind in population:
                fitness = self.func(ind)
                fitnesses.append(fitness)
            # Select the fittest individuals to refine the strategy
            fittest = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)[:self.budget]
            # Refine the strategy using the fittest individuals
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(fittest, 2)
                child = (parent1 + 2 * parent2) / 3
                new_population.append(child)
            population = new_population
        return population, fitnesses

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# The algorithm uses a novel search strategy that involves changing the direction of the search
# at each iteration, which can lead to better performance in certain cases. This strategy is
# inspired by the "directional mutation" technique used in evolutionary algorithms.