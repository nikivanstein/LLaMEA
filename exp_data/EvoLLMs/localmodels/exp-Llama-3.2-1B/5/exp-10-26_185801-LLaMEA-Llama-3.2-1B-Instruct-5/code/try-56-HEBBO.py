# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import numpy as np
import random
from collections import deque

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

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

    def mutate(self, individual):
        if random.random() < 0.05:
            mutation_index = random.randint(0, self.dim - 1)
            self.search_space[mutation_index] = random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.05:
            crossover_index = random.randint(0, self.dim - 1)
            child = parent1[:crossover_index] + parent2[crossover_index:]
            return child
        return parent1

    def selection(self, population):
        fitness_values = [self.__call__(func) for func in population]
        sorted_indices = np.argsort(fitness_values)
        sorted_population = [population[i] for i in sorted_indices]
        return sorted_population

    def next_generation(self, population):
        next_population = deque()
        for _ in range(self.budget):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_population.append(child)
        return next_population

    def optimize(self, func):
        population = [self.__call__(func)]
        while len(population) < 10:
            next_population = self.next_generation(population)
            population = next_population
        return self.__call__(func)

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolutionary Strategies
# 
# This algorithm uses an evolutionary strategy to optimize a black box function by iteratively refining the solution through mutation, crossover, and selection.