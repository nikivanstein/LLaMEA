# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def f(individual):
            return self.func(individual)
        
        # Define the bounds for the search space
        bounds = [(None, None) for _ in range(self.dim)]
        bounds[0] = (-5.0, -5.0)
        bounds[1] = (5.0, 5.0)

        # Perform a random search
        while True:
            for _ in range(self.budget):
                x = np.array(random.uniform(bounds[0][0], bounds[0][1]), dtype=float)
                if np.linalg.norm(f(x)) < self.budget / 2:
                    return x
            x = np.array(random.uniform(bounds[0][0], bounds[0][1]), dtype=float)
            bounds[0][0] = min(bounds[0][0], x[0])
            bounds[0][1] = max(bounds[0][1], x[0])
            bounds[1][0] = min(bounds[1][0], x[1])
            bounds[1][1] = max(bounds[1][1], x[1])

        # Refine the strategy
        while True:
            new_individual = np.array(random.uniform(bounds[0][0], bounds[0][1]), dtype=float)
            if np.linalg.norm(f(new_individual)) < self.budget / 2:
                return new_individual
            new_individual = np.array(random.uniform(bounds[0][0], bounds[0][1]), dtype=float)
            bounds[0][0] = min(bounds[0][0], new_individual[0])
            bounds[0][1] = max(bounds[0][1], new_individual[0])
            bounds[1][0] = min(bounds[1][0], new_individual[1])
            bounds[1][1] = max(bounds[1][1], new_individual[1])

class MetaHeuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BBOBOptimizer(budget, dim)

    def __call__(self, func):
        return self.optimizer(func)

# One-line description
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses a novel metaheuristic approach to optimize black box functions, 
# leveraging random search and bounds refinement to improve convergence rates.