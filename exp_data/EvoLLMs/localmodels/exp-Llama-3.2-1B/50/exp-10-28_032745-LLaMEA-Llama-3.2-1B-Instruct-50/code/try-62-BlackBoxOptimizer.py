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
            initial_guess = best_x
        return best_x, best_value

    def __new__(cls, initial_guess, iterations, budget):
        new_individual = self.evaluate_fitness(new_individual = initial_guess, iterations = iterations, budget = budget)
        return new_individual

    def evaluate_fitness(self, initial_guess, iterations, budget):
        # Novel Metaheuristic Algorithm for Black Box Optimization
        # 
        # Refine the strategy by changing the individual lines of the selected solution
        # 
        # Probability of change: 0.45
        # 
        # New fitness function: f_new(x) = f(x) + 0.1 * (x - x_new) * (x - x_old)
        # 
        # Update x_old and x_new at each iteration
        x_old = initial_guess
        x_new = initial_guess
        for i in range(iterations):
            new_x = [x + random.uniform(-0.01, 0.01) for x in x_new]
            new_value = self.func(new_x)
            if new_value < self.func(x_new):
                x_new = new_x
            if new_value > self.func(x_new):
                x_old = new_x
        return x_old, x_new

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Probability of change: 0.45
# 
# New fitness function: f_new(x) = f(x) + 0.1 * (x - x_new) * (x - x_old)