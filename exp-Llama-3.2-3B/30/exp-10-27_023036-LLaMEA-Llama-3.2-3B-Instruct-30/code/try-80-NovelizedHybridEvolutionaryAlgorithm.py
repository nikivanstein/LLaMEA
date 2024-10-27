import numpy as np
import random
import operator

class NovelizedHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.adaptation_rate = 0.3

    def initialize_population(self):
        return [self.sample_uniform() for _ in range(self.population_size)]

    def sample_uniform(self):
        return np.random.uniform(-5.0, 5.0, self.dim)

    def fitness(self, func, x):
        return func(x)

    def evaluate(self, func):
        self.population = self.evaluate_population(func)
        if self.budget > 0:
            self.population = self.adapt_and_evolve(func)

    def evaluate_population(self, func):
        population = [self.fitness(func, x) for x in self.population]
        return np.array(population)

    def adapt_and_evolve(self, func):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child = np.mean([parent1, parent2], axis=0)
            return child
        else:
            return parent1

    def mutate(self, x):
        if random.random() < self.mutation_rate:
            mutation = np.random.uniform(-1.0, 1.0, self.dim)
            x += mutation
            x = np.clip(x, -5.0, 5.0)
            return x
        else:
            return x

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.evaluate_population(func))

# Usage:
novelized_hhea = NovelizedHybridEvolutionaryAlgorithm(budget=100, dim=10)
result = novelized_hhea("noiseless_function_1")
print(result)