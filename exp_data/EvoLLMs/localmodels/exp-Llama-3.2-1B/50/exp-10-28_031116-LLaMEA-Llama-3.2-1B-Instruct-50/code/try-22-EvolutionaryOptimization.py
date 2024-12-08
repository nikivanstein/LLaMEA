import numpy as np
from scipy.optimize import differential_evolution
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            dim_values = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(dim_values)
        return population

    def __call__(self, func):
        while len(self.population) < self.budget:
            dim_values = random.sample(self.population, self.dim)
            func_values = func(dim_values)
            min_func_value = np.min(func_values)
            max_func_value = np.max(func_values)
            if min_func_value >= 0 and max_func_value <= 1:
                return dim_values
        return self.population[-1]

    def differential_evolution(self, func, bounds):
        result = differential_evolution(func, bounds, x0=self.population)
        return result.x

    def evaluate(self, func, dim_values):
        return func(dim_values)

    def run(self, func):
        best_func = None
        best_score = -np.inf
        for _ in range(self.budget):
            func_values = self.evaluate(func, self.__call__(func))
            if best_func is None or self.evaluate(func, best_func) > func_values:
                best_func = func_values
                best_score = self.evaluate(func, best_func)
        print(f"Best function: {best_func}, Score: {best_score}")
        return best_func, best_score

# One-line description with the main idea
# Evolutionary Optimization Algorithm using Differential Evolution (DE)
# This algorithm optimizes black box functions using a population-based approach and differential evolution optimization technique