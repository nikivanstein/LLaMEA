# Description: Novel Black Box Optimization using Genetic Algorithm with Evolved Mutation
# Code: 
# ```python
import numpy as np
import random

class GeneticAlgorithm:
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
        mutated_individual = individual.copy()
        if random.random() < 0.05:
            mutated_individual[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        if random.random() < 0.5:
            child[random.randint(0, self.dim-1)] = parent2[random.randint(0, self.dim-1)]
        return child

    def __next_generation(self, parent):
        child = parent.copy()
        if random.random() < 0.05:
            child = self.mutate(child)
        if random.random() < 0.05:
            child = self.crossover(child, parent)
        return child

    def optimize(self, func, num_generations):
        population = [self.__call__(func) for _ in range(100)]
        for generation in range(num_generations):
            new_population = []
            for _ in range(len(population)):
                parent1, parent2 = random.sample(population, 2)
                child = self.__next_generation(parent1, parent2)
                new_population.append(child)
            population = new_population
        return max(population)

# Description: Novel Black Box Optimization using Genetic Algorithm with Evolved Mutation
# Code: 
# ```python
ga = GeneticAlgorithm(budget=100, dim=5)
print(ga.optimize(func, num_generations=100))