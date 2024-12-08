import numpy as np
import random
import operator

class EvolutionaryBBOOPEvolutionStrategy:
    def __init__(self, budget, dim, mutation_rate, crossover_rate, selection_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_rate = selection_rate
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(100):
            individual = random.uniform(self.search_space)
            population.append(individual)
        return population

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if np.isnan(func(self.search_space)) or np.isinf(func(self.search_space)):
                raise ValueError("Invalid function value")
            if func(self.search_space) < 0 or func(self.search_space) > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func(self.search_space)

    def select_individual(self, population, selection_rate):
        sorted_indices = np.argsort(population)
        selected_indices = sorted_indices[:int(selection_rate * len(population))]
        selected_individuals = [population[i] for i in selected_indices]
        return selected_individuals

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child = parent1[:self.dim // 2] + parent2[self.dim // 2:]
            return child
        else:
            return parent1

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(self.dim), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def evaluate_fitness(self, individual, func):
        return func(individual)

    def update_population(self, population):
        selected_individuals = self.select_individual(population, self.selection_rate)
        new_population = []
        for _ in range(self.budget):
            parent1 = random.choice(selected_individuals)
            parent2 = random.choice(selected_individuals)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

# One-line description with the main idea
# Evolutionary Black Box Optimization using Evolution Strategies