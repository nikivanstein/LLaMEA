import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population = self.initialize_population()
        self.fitness_values = []

    def initialize_population(self):
        population = []
        for _ in range(100):  # initial population size
            individual = self.generate_individual()
            population.append(individual)
        return population

    def generate_individual(self):
        return [random.uniform(-5.0, 5.0) for _ in range(self.dim)]

    def evaluate_fitness(self, individual, func):
        return func(individual)

    def __call__(self, func):
        while len(self.fitness_values) < self.budget:
            fitness_values = []
            for individual in self.population:
                fitness_value = self.evaluate_fitness(individual, func)
                fitness_values.append(fitness_value)
            fitness_values.sort(reverse=True)
            selected_indices = [i for i, value in enumerate(fitness_values) if value >= self.search_space[0] and value <= self.search_space[1]]
            selected_individuals = [self.population[i] for i in selected_indices]
            new_individuals = []
            for _ in range(self.dim):
                parent1, parent2 = random.sample(selected_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.5:
                    child = parent1
                new_individuals.append(child)
            self.population = new_individuals
            self.fitness_values = [self.evaluate_fitness(individual, func) for individual in self.population]
        return self.population[0]

    def mutate(self, individual):
        if random.random() < 0.01:
            index1, index2 = random.sample(range(self.dim), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def crossover(self, parent1, parent2):
        child = (parent1 + parent2) / 2
        if random.random() < 0.5:
            child = parent1
        return child

    def selection(self):
        self.population.sort(key=self.evaluate_fitness, reverse=True)
        return self.population[0]

# Description: Evolutionary Black Box Optimization using Genetic Algorithm
# Code: 