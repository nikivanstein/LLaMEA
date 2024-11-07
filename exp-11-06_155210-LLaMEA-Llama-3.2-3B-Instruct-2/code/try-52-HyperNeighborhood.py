# import random
# import numpy as np
# import copy

# class HyperNeighborhood:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population = []
#         self.fitness_values = []
#         self.highest_fitness = -np.inf
#         self.best_solution = None
#         self.highest_fitness_index = -1

#     def __call__(self, func):
#         # Initialize the population with random solutions
#         for _ in range(self.budget):
#             solution = np.random.uniform(-5.0, 5.0, self.dim)
#             self.population.append(solution)
#             self.fitness_values.append(func(solution))

#         # Initialize the best solution and its fitness value
#         self.highest_fitness = max(self.fitness_values)
#         self.best_solution = self.population[self.fitness_values.index(self.highest_fitness)]

#         # Iterate until the budget is exhausted
#         for _ in range(self.budget):
#             # Select the best solution and its fitness value
#             for i in range(self.budget):
#                 if self.fitness_values[i] > self.highest_fitness:
#                     self.highest_fitness = self.fitness_values[i]
#                     self.best_solution = self.population[i]

#             # Create a new solution by perturbing the best solution
#             new_solution = copy.deepcopy(self.best_solution)
#             for i in range(self.dim):
#                 if np.random.rand() < 0.1:
#                     new_solution[i] += np.random.uniform(-0.1, 0.1)

#             # Evaluate the new solution
#             new_fitness = func(new_solution)

#             # Replace the best solution if the new solution is better
#             if new_fitness < self.highest_fitness:
#                 self.highest_fitness = new_fitness
#                 self.best_solution = new_solution

#             # Replace the least fit solution with the new solution
#             self.fitness_values[self.fitness_values.index(min(self.fitness_values))] = new_fitness
#             self.population[self.fitness_values.index(min(self.fitness_values))] = new_solution
# ```