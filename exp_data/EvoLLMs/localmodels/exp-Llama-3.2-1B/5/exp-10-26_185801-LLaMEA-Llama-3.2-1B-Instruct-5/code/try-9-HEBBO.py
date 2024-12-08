import numpy as np
import random
import copy

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = []

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

    def generate_population(self):
        while len(self.population) < self.population_size:
            individual = np.random.uniform(self.search_space)
            self.population.append(copy.deepcopy(individual))

    def select_parents(self, population):
        parents = []
        for _ in range(self.population_size // 2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            while parent1 == parent2:
                parent2 = random.choice(population)
            parent1_fitness = self.func(parent1)
            parent2_fitness = self.func(parent2)
            if random.random() < 0.5:
                parents.append((parent1, parent2, parent1_fitness, parent2_fitness))
            else:
                parents.append((parent2, parent1, parent2_fitness, parent1_fitness))
        return parents

    def crossover(self, parents):
        offspring = []
        for _ in range(self.population_size // 2):
            parent1, parent2, fitness1, fitness2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            fitness = self.func(child)
            if random.random() < 0.5:
                offspring.append((child, fitness))
            else:
                offspring.append((parent1, fitness))
        return offspring

    def mutate(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual[0] + random.uniform(-1, 1) / 10
            mutated_individual = np.clip(mutated_individual, self.search_space[0], self.search_space[1])
            mutated_individual_fitness = self.func(mutated_individual)
            mutated_offspring.append((mutated_individual, mutated_individual_fitness))
        return mutated_offspring

    def evaluate_fitness(self, individual, logger):
        func_value = self.func(individual)
        updated_individual = self.f(individual, logger)
        logger.update(updated_individual)
        return updated_individual, func_value

    def run(self, func):
        self.generate_population()
        parents = self.select_parents(self.population)
        offspring = self.crossover(parents)
        mutated_offspring = self.mutate(offspring)
        for individual, fitness in mutated_offspring:
            self.population.append(individual)
        while len(self.population) > self.budget:
            new_individual = self.evaluate_fitness(self.population[0], logger)
            logger.update(new_individual)
            self.population = [new_individual]
        return self.population[0]

# Description: Evolutionary Black Box Optimization using Adaptive Strategies
# Code: 