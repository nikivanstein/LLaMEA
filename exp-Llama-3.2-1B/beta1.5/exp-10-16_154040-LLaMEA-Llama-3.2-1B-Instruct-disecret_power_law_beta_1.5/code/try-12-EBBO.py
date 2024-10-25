import random
import numpy as np
from scipy.optimize import minimize

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
        dim = self.dim
        for _ in range(random.randint(0, self.dim)):
            new_dim = random.uniform(-5.0, 5.0)
            new_individual = individual[:dim] + [new_dim] + individual[dim:]
            individual = new_individual
        return individual

    def crossover(self, parent1, parent2):
        dim = self.dim
        child = parent1[:dim] + parent2[dim:]
        return child

    def selection(self, population):
        fitness = [self.evaluate(func) for func, _, _ in population]
        return np.random.choice(len(population), size=len(population), p=fitness)

    def next_generation(self, population):
        next_population = population.copy()
        for _ in range(self.budget):
            parents = self.selection(population)
            children = []
            for _ in range(self.population_size):
                if random.random() < 0.5:
                    parent1, parent2 = random.sample(parents, 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    children.append(child)
                else:
                    children.append(next_population[_])
            next_population = self.evaluate_fitness(children)
        return next_population

    def evaluate_fitness(self, fitness):
        return fitness

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm

# Code: Evolutionary Black Box Optimization using Genetic Algorithm