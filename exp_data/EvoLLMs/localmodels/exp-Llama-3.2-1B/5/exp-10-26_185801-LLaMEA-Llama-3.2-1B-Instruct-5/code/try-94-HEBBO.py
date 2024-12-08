import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = 0.1

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

class HeterogeneousEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = 0.1
        self.population_size = 100

    def __call__(self, func):
        population = [HEBBO(self.budget, self.dim) for _ in range(self.population_size)]
        while self.func_evaluations < self.budget:
            population = self.select(population, func)
            population = self.mutate(population)
            population = self.evaluate_fitness(population, func)
        return population[0]

    def select(self, population, func):
        selected_population = []
        for _ in range(self.population_size):
            new_individual = random.choice(population)
            if np.isnan(new_individual.search_space) or np.isinf(new_individual.search_space):
                new_individual = HEBBO(self.budget, self.dim)
            if new_individual.search_space not in selected_population:
                selected_population.append(new_individual)
        return selected_population

    def mutate(self, population):
        for individual in population:
            if random.random() < self.mutation_rate:
                new_individual = HEBBO(self.budget, self.dim)
                new_individual.search_space = np.linspace(-5.0, 5.0, self.dim)
                new_individual.func_evaluations = 0
                new_individual.search_space = individual.search_space
                new_individual.func_evaluations = 0
        return population

    def evaluate_fitness(self, population, func):
        fitnesses = []
        for individual in population:
            fitness = func(individual.search_space)
            fitnesses.append(fitness)
        return np.array(fitnesses)

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 