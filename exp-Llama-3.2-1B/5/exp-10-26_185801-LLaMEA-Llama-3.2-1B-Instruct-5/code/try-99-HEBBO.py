import numpy as np
import random
import operator
from collections import deque

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = deque(maxlen=self.population_size)

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

    def evaluate_fitness(self, individual):
        fitness = self.__call__(individual)
        return fitness

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim - 1)
            self.search_space[idx] = random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            idx = random.randint(0, self.dim - 1)
            child = parent1[:idx] + parent2[idx:]
            return child
        else:
            child = parent1
            return child

    def selection(self):
        fitnesses = [self.evaluate_fitness(individual) for individual in self.population]
        selected_indices = np.argsort(fitnesses)[-self.population_size:]
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def next_generation(self):
        population = self.population.copy()
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            population.append(child)
        population = deque(population)
        return population

    def run(self):
        while True:
            population = self.next_generation()
            fitnesses = [self.evaluate_fitness(individual) for individual in population]
            selected_indices = np.argsort(fitnesses)[-self.population_size:]
            selected_individuals = [self.population[i] for i in selected_indices]
            self.population = deque(selected_individuals)
            if np.mean(fitnesses) > 0.5:
                break
        return selected_individuals[0]

# One-line description with the main idea
# Evolutionary Black Box Optimization using Genetic Algorithm with Evolutionary Strategies
# Selects an individual from a population and evolves it through crossover and mutation to find the best solution
# The algorithm runs for a fixed number of iterations until a fixed number of evaluations are reached
# The fitness of each individual is evaluated and the best solution is selected