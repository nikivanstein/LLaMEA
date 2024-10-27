import numpy as np
from scipy.optimize import minimize
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        if random.random() < 0.05:
            dim = self.dim
            if random.random() < 0.5:
                dim -= 1
            if random.random() < 0.5:
                dim += 1
            self.search_space = np.linspace(-5.0, 5.0, dim)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            dim = self.dim
            if random.random() < 0.5:
                dim -= 1
            if random.random() < 0.5:
                dim += 1
            child = np.linspace(-5.0, 5.0, dim)
            child[:len(parent1)] = parent1
            child[len(parent1):] = parent2
            return child
        return parent1, parent2

    def evaluateBBOB(self, func, population_size, num_generations):
        for _ in range(num_generations):
            population = [self.__call__(func) for _ in range(population_size)]
            fitness = np.array([self.__call__(func) for func in population])
            best_individual = np.argmax(fitness)
            new_individual = random.choice(population)
            new_individual = self.mutate(new_individual)
            new_individual = self.crossover(new_individual, new_individual)
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                break
            fitness = np.array([self.__call__(func) for func in [new_individual, new_individual]])
            best_individual = np.argmax(fitness)
            if fitness[best_individual] > fitness[best_individual + 1]:
                best_individual += 1
        return best_individual

# Description: Novel metaheuristic algorithm for solving black box optimization problems using a combination of evolutionary and gradient-based search.
# Code: 