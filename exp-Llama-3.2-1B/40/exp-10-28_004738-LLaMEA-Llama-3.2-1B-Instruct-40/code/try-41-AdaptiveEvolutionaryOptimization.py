import random
import numpy as np

class AdaptiveEvolutionaryOptimization:
    def __init__(self, budget, dim, hyper_params):
        self.budget = budget
        self.dim = dim
        self.hyper_params = hyper_params
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, iterations):
        population = [x0]
        for _ in range(iterations):
            for _ in range(self.budget):
                new_individual = func(population[-1], self.hyper_params, bounds)
                if new_individual < bounds[0]:
                    new_individual = bounds[0]
                elif new_individual > bounds[1]:
                    new_individual = bounds[1]
                if random.random() < 0.4:
                    new_individual = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.2:
                    new_individual = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.4:
                    new_individual = random.uniform(bounds[0], bounds[1])
                population.append(new_individual)
            population = self._select(population)
            population = self._crossover(population)
            population = self._mutate(population)
        return population[-1]

    def _select(self, population):
        selection = random.choices(population, weights=self.hyper_params, k=len(population))
        return [individual for individual, weight in zip(population, selection) if weight > 0.5]

    def _crossover(self, population):
        offspring = []
        for _ in range(len(population) // 2):
            parent1, parent2 = random.sample(population, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def _mutate(self, population):
        for individual in population:
            if random.random() < 0.2:
                index1, index2 = random.sample(range(len(individual)), 2)
                individual[index1], individual[index2] = individual[index2], individual[index1]
        return population

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, iterations, hyper_params):
    return AdaptiveEvolutionaryOptimization(budget=iterations, dim=hyper_params, hyper_params=hyper_params).__call__(func, x0, bounds, iterations)

# Description: Adaptive Evolutionary Optimization using Hyper-Parametric Search
# Code: 