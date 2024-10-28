import random
import numpy as np
from scipy.optimize import differential_evolution

class EvolutionaryBlackBoxOptimization:
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
        dim = self.dim
        new_individual = individual.copy()
        new_individual[dim] += random.uniform(-1.0, 1.0)
        return new_individual

    def crossover(self, parent1, parent2):
        dim = self.dim
        child = parent1.copy()
        child[dim] = parent2[dim]
        return child

    def selection(self, population):
        fitness = np.array([self.evaluate(func) for func, _, _ in population])
        return fitness.argsort()[:-1].tolist()

    def mutate_selection(self, population, mutation_rate):
        fitness = np.array([self.evaluate(func) for func, _, _ in population])
        new_population = population.copy()
        for _ in range(self.population_size):
            idx = self.selection(population)
            new_individual = new_population.pop(idx)
            if random.random() < mutation_rate:
                new_individual = self.mutate(new_individual)
            new_population.append(new_individual)
        return new_population

    def evolve(self, population, mutation_rate):
        while len(population) > 0:
            new_population = self.mutate_selection(population, mutation_rate)
            fitness = np.array([self.evaluate(func) for func, _, _ in new_population])
            population = np.array(new_population).reshape(-1, self.dim).T
            population = self.selection(population)
            population = self.mutate_selection(population, mutation_rate)
        return population

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm