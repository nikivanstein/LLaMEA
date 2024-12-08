import random
import numpy as np

class EBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.mutation_rate = 0.01
        self mutation_threshold = 1.0
        self.population_fitness = np.zeros((self.population_size, self.dim))
        self.population_best = self.initialize_population_best()

    def initialize_population_best(self):
        return self.initialize_population()

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
        self.population_fitness = np.array([self.evaluate_fitness(individual) for individual in self.population])
        return func, self.population_fitness[np.argmax(self.population_fitness)]

    def evaluate_fitness(self, func):
        return func(func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            dim = self.dim * random.random()
            new_individual = self.generate_func(dim)
            return new_individual, new_individual, new_individual
        return individual, individual, individual

    def __str__(self):
        return "EBBO using Genetic Algorithm"

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm