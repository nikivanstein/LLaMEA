import random
import numpy as np
from scipy.optimize import differential_evolution

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
        new_individual = individual.copy()
        if random.random() < 0.1:
            dim = random.randint(1, self.dim)
            new_individual[dim] = random.uniform(-5.0, 5.0)
        return new_individual

    def crossover(self, parent1, parent2):
        dim = self.dim * random.random()
        child = parent1[:dim]
        child.extend(parent2[dim:])
        return child

    def optimize(self, func):
        population = self.population
        while len(population) > 0:
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                new_individual = self.mutate(child)
                new_population.append(new_individual)
            population = new_population
        return max(population, key=self.evaluate)

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm with Adaptive Mutation