import numpy as np
import random
import operator
import math

class GeneticProgramming:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select parents using tournament selection
            parents = self.select_parents()
            # Create offspring using crossover and mutation
            offspring = self.crossover_and_mutate(parents)
            # Evaluate fitness of offspring
            fitness = [self.evaluate_fitness(offspring[i], func) for i in range(self.population_size)]
            # Select fittest individual
            self.fittest_individual = self.select_fittest(offspring, fitness)
            # Update population
            self.population = self.population + [self.fittest_individual]
            self.func_evaluations += 1
        return self.fittest_individual

    def select_parents(self):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size):
            func_value = random.uniform(0, 1)
            for _ in range(self.dim):
                func_value = random.uniform(-5.0, 5.0)
                if func_value < 0 or func_value > 1:
                    raise ValueError("Function value must be between 0 and 1")
                func_value = func_value / 10.0
            parents.append((func_value, random.uniform(-5.0, 5.0)))
        # Select fittest individual
        parents.sort(key=operator.itemgetter(0), reverse=True)
        return parents

    def crossover_and_mutate(self, parents):
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < self.mutation_rate:
                func1, func2 = parent1
                func2, func3 = parent2
                func1 = func1 + random.uniform(-1.0, 1.0)
                func2 = func2 + random.uniform(-1.0, 1.0)
                func3 = func3 + random.uniform(-1.0, 1.0)
            else:
                func1, func2 = parent1
                func3 = parent2
            offspring.append((func1, func2, func3))
        return offspring

    def evaluate_fitness(self, individual, func):
        func_value = func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

# Description: Genetic Programming Algorithm for Black Box Optimization
# Code: 