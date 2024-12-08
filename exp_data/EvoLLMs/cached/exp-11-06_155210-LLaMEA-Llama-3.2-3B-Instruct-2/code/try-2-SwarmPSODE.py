# import numpy as np
# import random
# import operator

# class SwarmPSODE:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 50
#         self.swarm = np.zeros((self.population_size, self.dim))
#         self.vortex = np.zeros((self.population_size, self.dim))
#         self.mutation_rate = 0.2
#         self.c1 = 1.5
#         self.c2 = 1.5
#         self.w = 0.8

#     def __call__(self, func):
#         for _ in range(self.budget):
#             for i in range(self.population_size):
#                 self.swarm[i] = self.swarm[i] + self.c1 * random.random() * (self.swarm[i] - self.vortex[i]) + self.c2 * random.random() * (self.vortex[i] - self.swarm[(i+1)%self.population_size])
#                 self.swarm[i] = self.swarm[i] + self.w * self.swarm[i] + self.mutation_rate * (func(self.swarm[i]) - func(self.swarm[(i+1)%self.population_size]))
#                 self.vortex[i] = self.swarm[i]

#             # Update swarm
#             self.swarm = np.sort(self.swarm, axis=0)
#             self.swarm = np.vstack((self.swarm, np.zeros(self.dim)))
#             self.swarm = self.swarm[:self.population_size]

#         # Return best individual
#         best_idx = np.argmin([func(ind) for ind in self.swarm])
#         return self.swarm[best_idx]

# # Test the algorithm
# bbo_functions = [lambda x: x[0]**2 + 2*x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + 2*x[1]**2, lambda x: x[0]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + 2*x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + 2*x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + 2*x[1]**2, lambda x: x[0]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + x[1]**2, lambda x: x[0]**2 + 2*x[1]**2]
# swarm_pso_de = SwarmPSODE(50, 2)
# for i in range(len(bbo_functions)):
#     print(f"Function {i+1}: {bbo_functions[i]}")
#     print(f"Best solution: {swarm_pso_de(bbo_functions[i])}")
#     print()