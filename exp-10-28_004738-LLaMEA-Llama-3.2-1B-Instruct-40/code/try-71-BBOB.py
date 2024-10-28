import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        def evaluate_fitness(individual):
            return func(individual)

        def mutate(individual):
            if random.random() < 0.4:
                return individual + np.random.uniform(-5.0, 5.0)
            else:
                return individual

        def crossover(parent1, parent2):
            child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
            return child

        def selection(population):
            return sorted(population, key=lambda individual: evaluate_fitness(individual))

        population = selection(population)
        for _ in range(budget):
            individual = population[np.random.randint(0, len(population))]
            individual = mutate(individual)
            individual = crossover(individual, individual)
            individual = mutate(individual)
            if evaluate_fitness(individual) > evaluate_fitness(x0):
                population[np.random.randint(0, len(population))] = individual

        return x0 + np.random.uniform(-5.0, 5.0)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    x = x0
    for _ in range(budget):
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < 0.5:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

# Description: Black Box Optimization using BBOB
# Code: 