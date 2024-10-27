# import numpy as np
# import random

# class HybridEvolutionaryAlgorithm:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 20
#         self.population = self.initialize_population()
#         self.adaptive_probabilities = [0.3] * self.population_size

#     def initialize_population(self):
#         population = []
#         for _ in range(self.population_size):
#             individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
#             population.append(individual)
#         return population

#     def evaluate(self, func):
#         fitness_values = [func(individual) for individual in self.population]
#         for i in range(self.population_size):
#             self.population[i] = [individual for individual, fitness in zip(self.population, fitness_values) if fitness == min(fitness_values)]
#             self.adaptive_probabilities[i] = min(1, self.adaptive_probabilities[i] + (1 - self.adaptive_probabilities[i]) * (1 - fitness_values[i] / min(fitness_values)))

#     def select_parents(self):
#         parents = []
#         for _ in range(self.population_size):
#             parent_index = np.random.choice(self.population_size, p=self.adaptive_probabilities)
#             parents.append(self.population[parent_index])
#         return parents

#     def crossover(self, parents):
#         offspring = []
#         for _ in range(self.population_size):
#             parent1, parent2 = np.random.choice(parents, size=2, replace=False)
#             child = [0.5 * (parent1[i] + parent2[i]) for i in range(self.dim)]
#             offspring.append(child)
#         return offspring

#     def mutate(self, offspring):
#         for i in range(self.dim):
#             if random.random() < 0.1:
#                 offspring[i] += random.uniform(-1.0, 1.0)

#     def optimize(self, func):
#         for _ in range(self.budget):
#             self.evaluate(func)
#             parents = self.select_parents()
#             offspring = self.crossover(parents)
#             self.mutate(offspring)
#             self.population = offspring

# def test_func(x):
#     return np.sum([i**2 for i in x])

# algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=5)
# algorithm.optimize(test_func)
# print(algorithm.population)