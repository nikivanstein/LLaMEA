import random
import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()
        self.fitness_scores = self.calculate_fitness_scores()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def initialize_population(self):
        population = []
        for _ in range(100):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def calculate_fitness_scores(self):
        return self.fitness_scores

    def __call__(self, func, individual):
        fitness = func(individual)
        if random.random() < 0.4:
            # Randomly select a new individual
            new_individual = self.generate_new_individual(func, individual)
            fitness = func(new_individual)
        elif random.random() < 0.8:
            # Adapt the individual based on the fitness score
            adaptation_factor = self.adaptation_factor(individual, fitness)
            new_individual = self.adapt_individual(individual, adaptation_factor)
            fitness = func(new_individual)
        else:
            # Use the current individual
            fitness = func(individual)
        self.population.append(new_individual)
        self.fitness_scores.append(fitness)
        return fitness

    def generate_new_individual(self, func, individual):
        new_individual = individual[:]
        for _ in range(self.dim):
            new_individual.append(random.uniform(-5.0, 5.0))
        return new_individual

    def adapt_individual(self, individual, fitness):
        # Adapt the individual based on the fitness score
        # This could be a simple heuristic, such as
        # - if the fitness is high, add a small random perturbation
        # - if the fitness is low, remove a small random element
        adapt_factor = 0.5 + (fitness / 100)
        new_individual = [x + random.uniform(-1, 1) for x in individual]
        return new_individual

    def adapt_population(self):
        # Adapt the population based on the fitness scores
        # This could be a simple heuristic, such as
        # - if the fitness score is high, replace the individual with a new one
        # - if the fitness score is low, remove the individual with the lowest fitness
        adapt_factor = 0.5 + (self.fitness_scores / 100)
        self.population = [self.generate_new_individual(func, individual) for individual in self.population]
        self.population.sort(key=lambda x: x[1])
        self.population = self.population[:self.budget]

    def mutate(self):
        # Mutate the population based on the probability
        # This could be a simple heuristic, such as
        # - if the probability is high, mutate the individual with a small random perturbation
        # - if the probability is low, remove the individual with the highest fitness
        mutate_factor = 0.2 + (random.random() / 100)
        if mutate_factor > 0.5:
            # Mutate the individual with a small random perturbation
            new_individual = [x + random.uniform(-1, 1) for x in self.population[0]]
            self.population[0] = new_individual
        return self.population

    def evaluate_fitness(self, func):
        return func(self.population[0])

# Description: Evolutionary Optimization using Adaptive Genetic Algorithm
# Code: 
# ```python
# One-Liner Description:
# Evolutionary Optimization using Adaptive Genetic Algorithm
# Code: 