import random
import numpy as np

class GeneticProgramming:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, mutation_rate, population_size):
        # Evaluate the fitness of the initial population
        fitnesses = [self.f(func, x0, bounds, mutation_rate, population_size) for func in self.funcs]
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitnesses)[-self.budget:]
        # Create a new population by breeding the fittest individuals
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = self.breed(parent1, parent2, bounds, mutation_rate)
            new_population.append(child)
        # Replace the old population with the new one
        self.funcs = np.concatenate((self.funcs, new_population))
        return self.f(func, x0, bounds, mutation_rate, population_size)

    def breed(self, parent1, parent2, bounds, mutation_rate):
        child = parent1.copy()
        for _ in range(self.dim):
            if random.random() < mutation_rate:
                child[np.random.randint(0, self.dim)] = bounds[np.random.randint(0, self.dim)]
        return child

    def f(self, func, x, bounds, mutation_rate, population_size):
        return func(x)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return GeneticProgramming(budget, len(bounds)).__call__(func, x0, bounds, 0.4, 100)