import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population = []

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            self.population.append([best_x, best_value])
            initial_guess = best_x
        return self.population

    def mutate(self, individual):
        mutated_individual = individual.copy()
        if random.random() < 0.45:
            mutated_individual[0] += random.uniform(-0.01, 0.01)
        if random.random() < 0.45:
            mutated_individual[1] += random.uniform(-0.01, 0.01)
        return mutated_individual

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using evolutionary strategies
# 