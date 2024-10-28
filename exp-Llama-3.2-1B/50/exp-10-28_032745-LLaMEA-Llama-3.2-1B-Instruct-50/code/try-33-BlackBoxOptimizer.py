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

    def novel_search(self, initial_guess, iterations):
        # Refine the search strategy based on the fitness values
        fitness_values = [self.func(x) for x in initial_guess]
        best_index = np.argmin(fitness_values)
        best_individual = initial_guess[best_index]
        best_fitness = fitness_values[best_index]

        # Apply the probability 0.45 to refine the strategy
        if np.random.rand() < 0.45:
            # Change the individual lines of the selected solution
            best_individual = [x + random.uniform(-0.01, 0.01) for x in best_individual]
            best_fitness = self.func(best_individual)

        return best_individual, best_fitness

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 