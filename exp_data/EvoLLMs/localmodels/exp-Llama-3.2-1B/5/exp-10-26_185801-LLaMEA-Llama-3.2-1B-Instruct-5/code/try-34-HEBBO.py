import numpy as np
from collections import deque
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = [self.initialize_individual() for _ in range(100)]  # Initialize population with random individuals

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def initialize_individual(self):
        return np.random.uniform(-5.0, 5.0, self.dim)

    def mutate(self, individual):
        idx = random.randint(0, self.dim - 1)
        individual[idx] = np.random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        child = np.concatenate((parent1[:idx], parent2[idx:]))
        return child

    def selection(self, individuals):
        return np.random.choice(len(individuals), size=self.budget, replace=False)

    def mutate_exp(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            idx = random.randint(0, self.dim - 1)
            individual[idx] = np.random.uniform(-5.0, 5.0)
        return individual

    def fitness(self, individual):
        func_value = self.func_evaluations(individual)
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def run(self):
        population = self.population
        while len(population) > 0:
            new_population = []
            for _ in range(self.budget):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate_exp(child, 0.1)
                new_population.append(child)
            population = new_population
        return self.fitness(population[0])

# One-line description: Novel metaheuristic algorithm for black box optimization using evolutionary strategies