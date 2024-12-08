import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            func = np.random.uniform(-5.0, 5.0, self.dim)
            score = self.evaluate_func(func)
            population.append((func, score))
        return population

    def evaluate_func(self, func):
        func_score = np.inf
        for func_val, score in self.population:
            func_score = min(func_score, self.evaluate_func(func_val))
        return func_score

    def __call__(self, func):
        for _ in range(self.budget):
            func_score = self.evaluate_func(func)
            if func_score < np.inf:
                return func
        return None

    def mutate(self, func):
        func_val = func.copy()
        if random.random() < 0.5:
            func_val += random.uniform(-1.0, 1.0)
        return func_val

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            func1, func2 = parent1
            func3, func4 = parent2
        else:
            func1, func2 = parent2
            func3, func4 = parent1
        child1 = (func1 + func2) / 2
        child2 = (func3 + func4) / 2
        return child1, child2

    def run(self):
        for _ in range(100):
            for func, score in self.population:
                new_func = self.__call__(func)
                if new_func is not None:
                    self.population.append((new_func, score))
            self.population = self.population[:self.population_size]
        return self.population

# One-line description with main idea
# Evolutionary Algorithm with Adaptive Crossover and Mutation
# The algorithm uses adaptive crossover and mutation to search for the optimal function in the population.

# Code