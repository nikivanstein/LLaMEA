import random
import numpy as np

class EvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()
        self.population_history = []

    def initialize_population(self):
        population = []
        for _ in range(100):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def fitness(self, individual, func):
        return func(individual)

    def crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutation(self, individual, func):
        if random.random() < 0.2:
            index = random.randint(0, self.dim - 1)
            individual[index] = func(np.random.uniform(-5.0, 5.0))
        return individual

    def selection(self, population):
        return np.array(population)

    def __call__(self, func, bounds, budget):
        individual = self.selection(self.population)
        population = self.population.copy()
        for _ in range(budget):
            new_individual = self.crossover(individual, individual)
            new_individual = self.mutation(new_individual, func)
            population.append(new_individual)
            individual = new_individual
            population.remove(individual)
            if len(population) > 0:
                population.sort(key=self.fitness, reverse=True)
                population = population[:self.budget]
        return population[0]

# Description: Evolutionary Algorithm using Evolution Strategy (EAES)
# Code: 
# ```python
# BBOB: Black Box Optimization using BBOB
# Code: 