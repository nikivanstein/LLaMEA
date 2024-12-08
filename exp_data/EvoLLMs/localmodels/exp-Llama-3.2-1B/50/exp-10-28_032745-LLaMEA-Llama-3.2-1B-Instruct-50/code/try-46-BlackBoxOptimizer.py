import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population_size = 100
        self.population = np.random.rand(self.population_size, self.dim)
        self.iterations = 1000
        self.step_size = 0.01
        self.fitness_history = []

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-self.step_size, self.step_size) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            self.population = [x for x in self.population if self.func(x) >= best_value]
            self.fitness_history.append(self.func(self.population))
            if len(self.fitness_history) > self.iterations:
                self.population = np.random.rand(self.population_size, self.dim)
            initial_guess = best_x
        return best_x, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using adaptive step size and evolutionary strategy