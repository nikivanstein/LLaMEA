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

        def mutate(individual):
            return [x + random.uniform(-0.01, 0.01) for x in individual]

        def mutate_multiple(individuals):
            return [mutate(individual) for individual in individuals]

        best_individual = initial_guess
        best_value = evaluate_fitness(best_individual)

        for _ in range(iterations):
            if _ >= self.budget:
                break
            for _ in range(self.dim):
                new_individual = mutate(best_individual)
                new_value = evaluate_fitness(new_individual)
                if new_value < best_value:
                    best_individual = new_individual
                    best_value = new_value
            best_individual = mutate_multiple([best_individual])
        return best_individual, best_value

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 