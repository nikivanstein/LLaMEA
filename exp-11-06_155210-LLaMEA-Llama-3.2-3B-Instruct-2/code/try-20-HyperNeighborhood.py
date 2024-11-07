# import random
# import numpy as np
# import functools

# class HyperNeighborhood:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population = []
#         self.fitness_values = []
#         self.cache = {}
#         self.fitness_cache = {}

#     def __call__(self, func):
#         # Initialize the population with random solutions
#         for _ in range(self.budget):
#             solution = np.random.uniform(-5.0, 5.0, self.dim)
#             self.population.append(solution)
#             self.fitness_values.append(func(solution))

#         # Initialize the best solution and its fitness value
#         best_solution = self.population[0]
#         best_fitness = self.fitness_values[0]

#         # Iterate until the budget is exhausted
#         for _ in range(self.budget):
#             # Select the best solution and its fitness value
#             for i in range(self.budget):
#                 if self.fitness_values[i] > best_fitness:
#                     best_solution = self.population[i]
#                     best_fitness = self.fitness_values[i]

#             # Create a new solution by perturbing the best solution
#             new_solution = best_solution + np.random.uniform(-0.1, 0.1, self.dim)

#             # Evaluate the new solution
#             new_fitness = func(new_solution)

#             # Replace the best solution if the new solution is better
#             if new_fitness < best_fitness:
#                 best_solution = new_solution
#                 best_fitness = new_fitness

#             # Replace the least fit solution with the new solution
#             self.fitness_values[self.fitness_values.index(min(self.fitness_values))] = new_fitness
#             self.population[self.fitness_values.index(min(self.fitness_values))] = new_solution

#             # Cache the fitness value to avoid redundant evaluations
#             self.cache[new_solution] = new_fitness
#             self.fitness_cache[new_solution] = new_fitness

#             # Check if the fitness value is already cached
#             if new_solution in self.fitness_cache:
#                 new_fitness = self.fitness_cache[new_solution]

#         # Remove cached solutions that are no longer in the population
#         for solution in list(self.cache.keys()):
#             if solution not in self.population:
#                 del self.cache[solution]
#                 del self.fitness_cache[solution]

#     def _get_fitness(self, solution):
#         # Check if the fitness value is cached
#         if solution in self.fitness_cache:
#             return self.fitness_cache[solution]

#         # Evaluate the solution and cache the result
#         fitness = func(solution)
#         self.fitness_cache[solution] = fitness
#         return fitness

# test_hyper_neighborhood()
# ```