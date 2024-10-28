import random
import numpy as np

class EBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()

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
        dim = self.dim * random.random()
        func = individual[0]
        new_individual = (func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        return new_individual

    def crossover(self, parent1, parent2):
        dim = self.dim * random.random()
        func1 = parent1[0]
        func2 = parent2[0]
        new_individual = (func1, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        return new_individual

    def selection(self, population):
        return sorted(population, key=lambda individual: individual[2], reverse=True)[:self.population_size // 2]

    def __str__(self):
        return f"EBBO using Genetic Algorithm with budget {self.budget} and dimension {self.dim}"

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm