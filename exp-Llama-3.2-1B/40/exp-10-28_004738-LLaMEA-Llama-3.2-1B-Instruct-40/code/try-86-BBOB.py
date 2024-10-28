import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.mutation_rate = 0.4
        self.population_size = 100
        self.population = self.initialize_population()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0, self.dim)
            functions.append(func)
        return functions

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            x0 = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(self.funcs[0](x0))
        return population

    def __call__(self, func):
        return func(np.random.uniform(-5.0, 5.0, self.dim))

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i = random.randint(0, self.dim - 1)
            individual[i] = np.random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            i = random.randint(0, self.dim - 1)
            j = random.randint(0, self.dim - 1)
            parent1[i], parent1[j] = parent1[j], parent1[i]
            parent2[i], parent2[j] = parent2[j], parent2[i]
        return parent1, parent2

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
    population = self.population
    for _ in range(budget):
        new_population = []
        for individual in population:
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        population = new_population
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
    return x

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization using Adaptive Mutation Rate
# Code: 
# ```python
# ```python
# ```python
# ```python
# ```python
# ```python