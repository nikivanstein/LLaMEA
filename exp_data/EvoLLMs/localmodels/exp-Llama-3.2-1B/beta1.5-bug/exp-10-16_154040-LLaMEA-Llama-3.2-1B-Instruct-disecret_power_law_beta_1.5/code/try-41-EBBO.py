# Description: Evolutionary Black Box Optimization using Genetic Algorithm
# Code: 
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
        def fitness(individual):
            return self.evaluate(func, individual)
        
        for _ in range(self.budget):
            fitness(individual)
            if fitness(individual) < 0:
                break
        return func, fitness(individual)

    def evaluate(self, func, individual):
        return func(func, individual, random.uniform(-5.0, 5.0))

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm

# Code:
import numpy as np
import random
import math

class EBO:
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
        def fitness(individual):
            return self.evaluate(func, individual)

        def mutate(individual):
            dim = self.dim * random.random()
            func = self.generate_func(dim)
            individual = np.append(individual, func)
            return individual

        def crossover(parent1, parent2):
            dim = self.dim * random.random()
            func1 = self.generate_func(dim)
            func2 = self.generate_func(dim)
            individual1 = np.append(parent1, func1)
            individual2 = np.append(parent2, func2)
            return individual1, individual2

        for _ in range(self.budget):
            individual = random.choice(self.population)
            if random.random() < 0.5:
                individual = mutate(individual)
            individual = self.crossover(individual, individual)
            fitness(individual)
            if fitness(individual) < 0:
                break
        return func, fitness(individual)

    def evaluate(self, func, individual):
        return func(func, individual, random.uniform(-5.0, 5.0))

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm