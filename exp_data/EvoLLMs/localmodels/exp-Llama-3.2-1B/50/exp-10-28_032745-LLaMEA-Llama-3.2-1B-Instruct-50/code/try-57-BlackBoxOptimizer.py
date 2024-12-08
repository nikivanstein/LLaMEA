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

        def new_individual(individual):
            new_individual = [x + random.uniform(-0.01, 0.01) for x in individual]
            return new_individual

        def fitness(individual):
            return evaluate_fitness(individual)

        def mutate(individual):
            new_individual = new_individual(individual)
            if random.random() < 0.45:
                new_individual = new_individual[::-1]
            return new_individual

        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_individual = initial_guess
            best_value = fitness(best_individual)
            for i in range(self.dim):
                new_individual = new_individual(best_individual)
                new_value = fitness(new_individual)
                if new_value < best_value:
                    best_individual = new_individual
                    best_value = new_value
            initial_guess = best_individual
        return best_individual, best_value

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# 0.45 probability of changing the individual's direction
# 0.55 probability of changing the individual's position
# 
# Example usage:
# optimizer = BlackBoxOptimizer(100, 2)
# best_individual, best_value = optimizer(func, [1, 1], 100)