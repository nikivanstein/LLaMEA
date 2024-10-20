import random
import numpy as np

class EBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.selection_rate = 0.5

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim * random.random()
            func = self.generate_func(dim)
            population.append((func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)))
        return population

    def generate_func(self, dim):
        return np.sin(np.sqrt(dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = self.evaluate(func)
            if fitness < 0:
                break
        return func, fitness

    def evaluate(self, func):
        return func(func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            dim = self.dim * random.random()
            func = self.generate_func(dim)
            new_individual = (func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
            return new_individual
        return individual

    def select(self, population):
        return random.choices(population, weights=self.selection_rate, k=self.population_size)

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            dim = self.dim * random.random()
            func1 = self.generate_func(dim)
            func2 = self.generate_func(dim)
            new_individual = (func1, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
            return new_individual
        return None

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm with Adaptive Mutation and Selection