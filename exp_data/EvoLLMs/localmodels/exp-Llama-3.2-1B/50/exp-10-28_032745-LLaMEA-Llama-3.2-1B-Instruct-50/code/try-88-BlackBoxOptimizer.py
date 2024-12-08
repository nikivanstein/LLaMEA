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
            best_x = copy.deepcopy(best_x)
            self.evaluate_fitness(best_x, self.logger)
            initial_guess = copy.deepcopy(best_x)
        return best_x, best_value

    def evaluate_fitness(self, individual, logger):
        updated_individual = individual
        for i in range(self.dim):
            new_individual = [x + random.uniform(-0.01, 0.01) for x in updated_individual]
            new_fitness = self.func(new_individual)
            if new_fitness < updated_individual[i] * 0.45 + updated_individual[i] * 0.55:
                updated_individual[i] *= 0.95
        logger.log('Fitness updated:', updated_individual, new_fitness)
        return updated_individual, new_fitness

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 