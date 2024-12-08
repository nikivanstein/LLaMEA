import numpy as np
from scipy.optimize import minimize
import random

class EBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_population()

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            bounds = [(-5.0, 5.0)] * dim
            func = lambda x: np.sum(x**2)
            res = minimize(func, np.array([0.0]*dim), method="SLSQP", bounds=bounds, options={"maxiter": 1000})
            population.append(res.x)
        return population

    def __call__(self, func):
        def optimize(func):
            def optimize_func(x):
                return func(x)
            return optimize_func
        return self.__call__(optimize)

    def select_solution(self, func, population):
        if len(population) < self.budget:
            return func
        else:
            idx = random.randint(0, len(population) - 1)
            return population[idx]

    def mutate(self, func, population):
        def mutate_func(x):
            return func(x)
        return mutate_func

    def crossover(self, func, population):
        def crossover_func(x1, x2):
            return func(x1) + func(x2)
        return crossover_func

    def mutate_and_crossover(self, func, population):
        def mutate_and_crossover_func(x1, x2):
            return func(x1) + func(x2)
        return mutate_and_crossover_func

    def update_population(self, func, population):
        for _ in range(self.budget):
            idx1 = self.select_solution(func, population)
            idx2 = self.select_solution(func, population)
            x1 = population[idx1]
            x2 = population[idx2]
            if random.random() < 0.45:
                x1 = self.mutate(func, population)
            if random.random() < 0.45:
                x2 = self.mutate(func, population)
            if random.random() < 0.45:
                x1 = self.crossover(func, population)
                x2 = self.crossover(func, population)
            population[idx1] = x1
            population[idx2] = x2