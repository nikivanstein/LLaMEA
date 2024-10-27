# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# import numpy as np
# import random
# import operator
# import copy
# import time

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.new_individuals = []

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
            new_individual = copy.deepcopy(individual)
            new_individual[self.search_space.index(min(new_individual))] = random.uniform(self.search_space[self.search_space.index(min(new_individual))], self.search_space[self.search_space.index(max(new_individual))])
            return new_individual
        else:
            return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.05:
            child = copy.deepcopy(parent1)
            child[self.search_space.index(min(child))] = parent2[self.search_space.index(min(parent2))]
            return child
        else:
            return parent1

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        return func_value

    def fitness(self, individual):
        func_value = self.evaluate_fitness(individual)
        return func_value

# Example usage
def func1(x):
    return np.sin(x)

def func2(x):
    return np.cos(x)

def func3(x):
    return np.exp(x)

hebbbo = HEBBO(10, 5)
print("Initial population:")
for i, individual in enumerate(hebbbo.new_individuals):
    print(f"Individual {i+1}: {individual}")

for i in range(10):
    for j in range(i+1):
        individual1 = hebbbo.new_individuals[i]
        individual2 = hebbbo.new_individuals[j]
        individual3 = hebbbo.new_individuals[i]
        print(f"\nIndividual 1: {individual1}")
        print(f"Individual 2: {individual2}")
        print(f"Individual 3: {individual3}")
        print(f"Fitness of Individual 1: {hebbbo.fitness(individual1)}")
        print(f"Fitness of Individual 2: {hebbbo.fitness(individual2)}")
        print(f"Fitness of Individual 3: {hebbbo.fitness(individual3)}")
        print("\n")

# Select the best individual
best_individual = min(hebbbo.new_individuals, key=hebbbo.fitness)
print(f"\nBest individual: {best_individual}")