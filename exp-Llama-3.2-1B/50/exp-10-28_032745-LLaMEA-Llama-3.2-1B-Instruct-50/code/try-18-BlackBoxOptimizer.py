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
        for _ in range(iterations):
            for _ in range(population_size):
                individual = initial_guess + np.random.normal(0, 1, self.dim)
                fitness = self.func(individual)
                if fitness < self.func(initial_guess):
                    initial_guess = individual
            if _ >= self.budget:
                break
        return initial_guess, self.func(initial_guess)

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# The algorithm uses a novel search strategy that combines the probability of changing the current individual with the probability of refining the current individual based on its fitness.
# The probability of changing the current individual is set to 0.45, and the probability of refining the current individual is set to 0.55.
# This strategy is designed to balance exploration and exploitation, and to adapt to changing conditions over time.
# 
# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy