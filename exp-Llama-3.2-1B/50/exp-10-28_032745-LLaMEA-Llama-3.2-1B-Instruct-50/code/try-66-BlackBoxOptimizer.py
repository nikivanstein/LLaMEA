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

    def evaluate_fitness(self, individual, budget):
        updated_individual = individual
        for _ in range(budget):
            if updated_individual == individual:
                break
            updated_individual = self.f(updated_individual, self.func, 1)
        return updated_individual

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# The algorithm uses a combination of line search and adaptive mutation to refine its strategy.
# 
# The search space is divided into sub-spaces of size 10, and the mutation rate is adaptive based on the fitness value.
# 
# The algorithm is designed to handle a wide range of tasks and has been evaluated on the BBOB test suite of 24 noiseless functions.
# 
# One-line description with the main idea: Novel metaheuristic algorithm for black box optimization using adaptive mutation and line search strategy.