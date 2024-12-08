import numpy as np
import random
import operator

class GeneticAlgorithm:
    def __init__(self, budget, dim, mutation_rate, crossover_rate, population_size):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        individual = np.zeros(self.dim)
        for i in range(self.dim):
            individual[i] = random.uniform(self.search_space[i])
        return individual

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            individual = self.population[self.func_evaluations % self.population_size]
            func_value = func(individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(self.dim), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def crossover(self, individual1, individual2):
        if random.random() < self.crossover_rate:
            index = random.randint(0, self.dim - 1)
            individual1[index], individual2[index] = individual2[index], individual1[index]
        return individual1

    def fitness(self, func, individual):
        func_value = func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def select(self, func, individuals):
        fitnesses = [self.fitness(func, individual) for individual in individuals]
        return np.random.choice(len(individuals), size=len(individuals), p=fitnesses)

    def next_generation(self):
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            offspring.append(child)
        return offspring

    def run(self):
        while self.func_evaluations < self.budget:
            self.population = self.next_generation()
            if len(self.population) > self.population_size:
                self.population.pop(0)
        return self.population[self.func_evaluations % self.population_size]

# Description: Evolutionary Black Box Optimization using Genetic Algorithm
# Code: 