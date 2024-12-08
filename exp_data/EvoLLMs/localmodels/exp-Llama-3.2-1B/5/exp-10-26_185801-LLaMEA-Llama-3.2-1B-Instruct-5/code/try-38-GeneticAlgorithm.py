import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    def create_individual(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.dim)]

    def evaluate_fitness(self, individual):
        func_value = self.func_evaluations(individual, self.budget)
        self.func_evaluations(individual, self.budget)
        return func_value

    def select_parents(self):
        fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
        parents = [individual for i, individual in enumerate(self.population) if fitness_scores[i] < 0.5]
        return parents

    def crossover(self, parent1, parent2):
        child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(len(individual)), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            parents = self.select_parents()
            children = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                children.append(child)
            self.population = children
        return self.evaluate_fitness(self.population[0])