import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.elite_size = 10
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.choice(self.search_space, size=self.dim, replace=False)
            population.append(individual)
        return population

    def __call__(self, func):
        while len(self.population) < self.budget:
            fitness_values = [self.evaluate_fitness(individual, func) for individual in self.population]
            self.population = self.select_fittest(population, fitness_values)
        return self.population[0]

    def select_fittest(self, population, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        sorted_population = population[sorted_indices]
        sorted_fitness_values = fitness_values[sorted_indices]
        elite = sorted_population[:self.elite_size]
        return elite

    def evaluate_fitness(self, individual, func):
        func_value = func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(self.dim), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            index = random.randint(0, self.dim - 1)
            parent1[index], parent2[index] = parent2[index], parent1[index]
        return parent1, parent2

# Description: Novel metaheuristic algorithm using Genetic Algorithm for Black Box Optimization
# Code: 