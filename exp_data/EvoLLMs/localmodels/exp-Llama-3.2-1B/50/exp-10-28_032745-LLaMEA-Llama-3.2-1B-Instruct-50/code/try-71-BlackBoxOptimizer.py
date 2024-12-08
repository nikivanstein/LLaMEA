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
        population = [initial_guess] * self.budget
        for _ in range(iterations):
            for i in range(self.budget):
                if random.random() < 0.45:
                    new_individual = population[i]
                    updated_value = func(new_individual)
                    if updated_value < population[i + 1][0]:
                        population[i + 1] = new_individual
        return population

# Novel Metaheuristic Algorithm for Black Box Optimization (BMBA)
# 
# The BMBA algorithm is inspired by the concept of "refining the search space" to refine the individual lines of the selected solution. This is achieved by introducing a probability of changing the individual line, which is 45% of the time, and 55% of the time, which is the original probability. This helps to escape local optima and converge to the global optimum.
# 
# The code below implements the BMBA algorithm.