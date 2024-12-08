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
        population = [initial_guess]
        for _ in range(iterations):
            if _ >= self.budget:
                break
            new_population = []
            for _ in range(self.dim):
                new_individual = copy.deepcopy(population[-1])
                for i in range(self.dim):
                    new_individual[i] += random.uniform(-0.01, 0.01)
                new_individual = [x + random.uniform(-0.01, 0.01) for x in new_individual]
                new_individual = self.evaluate_fitness(new_individual)
                new_population.append(new_individual)
            population = new_population
        return population[0], population[-1]

    def evaluate_fitness(self, individual):
        return self.func(individual)

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 