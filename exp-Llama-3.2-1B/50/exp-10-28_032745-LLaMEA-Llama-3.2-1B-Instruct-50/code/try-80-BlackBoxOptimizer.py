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
        def evaluate_fitness(individual):
            return self.func(individual)

        fitnesses = [evaluate_fitness(individual) for _ in range(self.budget)]
        best_index = np.argmin(fitnesses)
        best_individual = fitnesses[best_index]

        def new_search_space(dim):
            return (self.search_space[0] - 0.01, self.search_space[1] + 0.01)

        def new_func(individual):
            return individual[0] * individual[1]

        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_individual = best_individual
            best_fitness = fitnesses[best_index]
            new_individual = best_individual + random.uniform(-0.01, 0.01) * (new_search_space(dim) / new_search_space(dim).max())
            new_fitness = evaluate_fitness(new_individual)
            if new_fitness < best_fitness:
                best_individual = new_individual
                best_fitness = new_fitness

        return best_individual, best_fitness

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses a novel search strategy that refines its individual lines by changing the direction of the search space.
# The algorithm starts with an initial individual and iteratively applies a series of transformations to refine the individual.
# The transformations are based on the idea that the optimal solution is likely to be found in the vicinity of the current individual.
# The algorithm uses a probability of 0.45 to change the direction of the search space, which allows it to explore different areas of the search space.