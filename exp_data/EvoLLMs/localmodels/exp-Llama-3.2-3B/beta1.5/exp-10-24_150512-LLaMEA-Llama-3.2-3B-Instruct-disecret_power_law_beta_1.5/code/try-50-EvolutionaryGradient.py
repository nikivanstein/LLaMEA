# import numpy as np
# import random
# import time

# class EvolutionaryGradient:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.x = np.random.uniform(-5.0, 5.0, size=dim)
#         self.f_best = np.inf
#         self.x_best = None
#         self.population_size = 10
#         self.crossover_probability = 0.025
#         self.mutation_probability = 0.1

#     def __call__(self, func):
#         for _ in range(self.budget):
#             # Compute gradient of the objective function
#             gradient = np.zeros(self.dim)
#             h = 1e-1
#             for i in range(self.dim):
#                 gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

#             # Update the current solution using evolutionary strategy
#             self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)

#             # Select parents for crossover
#             parents = np.random.choice(self.population_size, size=2, replace=False)

#             # Perform crossover
#             if random.random() < self.crossover_probability:
#                 child = np.zeros(self.dim)
#                 for i in range(self.dim):
#                     if random.random() < 0.5:
#                         child[i] = self.x[parents[0], i]
#                     else:
#                         child[i] = self.x[parents[1], i]
#                 self.x = child

#             # Perform mutation
#             if random.random() < self.mutation_probability:
#                 for i in range(self.dim):
#                     self.x[i] += np.random.normal(0, 0.1)

#             # Update the best solution
#             f = func(self.x)
#             if f < self.f_best:
#                 self.f_best = f
#                 self.x_best = self.x.copy()

#             # Check for convergence
#             if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
#                 print("Converged after {} iterations".format(_))

# # Example usage:
# def func(x):
#     return np.sum(x**2)

# evg = EvolutionaryGradient(budget=1000, dim=10)
# evg("func")
# ```