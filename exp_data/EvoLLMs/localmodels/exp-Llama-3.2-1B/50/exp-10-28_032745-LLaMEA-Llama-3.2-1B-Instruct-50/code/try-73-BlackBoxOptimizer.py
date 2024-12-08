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
        
        def next_individual(individual):
            new_individual = individual
            for _ in range(self.dim):
                if random.random() < 0.45:  # Refine strategy
                    new_individual = [x + random.uniform(-0.01, 0.01) for x in new_individual]
            return new_individual
        
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = evaluate_fitness(best_x)
            for i in range(self.dim):
                new_x = next_individual(best_x)
                new_value = evaluate_fitness(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            initial_guess = best_x
        return best_x, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy