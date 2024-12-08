import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            new_individual = self.evaluate_fitness(func)
            self.population.append(new_individual)
            self.func_evaluations += 1
            if np.isnan(new_individual[0]) or np.isinf(new_individual[0]):
                raise ValueError("Invalid function value")
            if new_individual[0] < 0 or new_individual[0] > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return self.population

    def evaluate_fitness(self, func):
        return func(self.search_space)

    def mutate(self, individual):
        if np.random.rand() < 0.05:
            index = np.random.randint(0, self.dim)
            new_individual = self.search_space.copy()
            new_individual[index] = np.random.uniform(self.search_space[index] + 1, self.search_space[index] - 1)
            return new_individual
        else:
            return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.05:
            index = np.random.randint(0, self.dim)
            new_individual = self.search_space.copy()
            new_individual[index] = (parent1[index] + parent2[index]) / 2
            return new_individual
        else:
            return parent1, parent2

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using mutation and crossover
# 
# The algorithm uses mutation and crossover operators to refine the solution
# 
# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 