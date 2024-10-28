import random
import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim, adaptive_budget, adaptive_func):
        self.budget = budget
        self.dim = dim
        self.adaptive_budget = adaptive_budget
        self.adaptive_func = adaptive_func
        self.population = []
        self.population_history = []
        self.fitness_history = []

    def generate_population(self, size):
        self.population = [random.uniform(-5.0, 5.0) for _ in range(size)]

    def fitness(self, individual):
        return self.adaptive_func(individual)

    def mutate(self, individual):
        if random.random() < 0.1:
            return random.uniform(-5.0, 5.0)
        return individual

    def evaluate_fitness(self, individual):
        return self.fitness(individual)

    def __call__(self, func, bounds, budget):
        self.generate_population(budget)
        for _ in range(budget):
            individual = self.population[np.random.randint(0, len(self.population))]
            fitness = self.evaluate_fitness(individual)
            if fitness < bounds[0]:
                individual = bounds[0]
            elif fitness > bounds[1]:
                individual = bounds[1]
            if random.random() < 0.4:
                individual = self.adaptive_func(individual)
            if random.random() < 0.2:
                individual = self.adaptive_func(individual)
            if random.random() < 0.4:
                individual = self.adaptive_func(individual)
            self.population_history.append(individual)
            self.fitness_history.append(fitness)
        return individual

# Description: Evolutionary Optimization using Adaptative Genetic Algorithm
# Code: 