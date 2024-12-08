# import random
# import numpy as np
# import math

# class HyperNeighborhood:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population = []
#         self.fitness_values = []
#         self.neighborhood_size = 5

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

#             # Increase the neighborhood size for better exploration
#             if math.fabs(best_fitness - min(self.fitness_values)) < 0.1:
#                 self.neighborhood_size += 1
#             else:
#                 self.neighborhood_size = min(self.neighborhood_size, 10)

#     def get_neighborhood(self, solution):
#         neighborhood = []
#         for _ in range(self.neighborhood_size):
#             neighbor = solution + np.random.uniform(-0.1, 0.1, self.dim)
#             neighborhood.append(neighbor)
#         return neighborhood

# # Test the HyperNeighborhood algorithm
# def test_hyper_neighborhood():
#     func = lambda x: sum([i**2 for i in x])
#     hyper_neighborhood = HyperNeighborhood(100, 10)
#     hyper_neighborhood(func)

# test_hyper_neighborhood()
# ```