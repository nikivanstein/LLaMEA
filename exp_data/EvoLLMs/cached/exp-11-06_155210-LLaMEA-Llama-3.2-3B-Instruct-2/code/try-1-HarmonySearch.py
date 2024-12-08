# import numpy as np
# import random
# import math

# class HarmonySearch:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 50
#         self.crossover_rate = 0.8
#         self.mutation_rate = 0.1
#         self.harmony_memory = []
#         self.best_solution = None

#     def __call__(self, func):
#         # Initialize the harmony memory
#         for _ in range(self.population_size):
#             x = np.random.uniform(-5.0, 5.0, self.dim)
#             f = func(x)
#             self.harmony_memory.append((x, f))

#         # Select the best solution
#         self.best_solution = min(self.harmony_memory, key=lambda x: x[1])

#         # Perform the harmony search
#         while len(self.harmony_memory) < self.budget:
#             # Create a new harmony
#             x = np.random.uniform(-5.0, 5.0, self.dim)
#             f = func(x)
#             self.harmony_memory.append((x, f))

#             # Update the harmony memory
#             self.harmony_memory.sort(key=lambda x: x[1])

#             # Select the best solution
#             self.best_solution = min(self.harmony_memory, key=lambda x: x[1])

#             # Perform crossover and mutation
#             if random.random() < self.crossover_rate:
#                 # Perform crossover
#                 x1, x2 = random.sample(self.harmony_memory, 2)
#                 x = (x1[0] + x2[0]) / 2
#             else:
#                 # Perform mutation
#                 x = x + random.uniform(-0.1, 0.1, self.dim)

#             # Evaluate the new solution
#             f = func(x)

#             # Update the harmony memory
#             self.harmony_memory.remove((x, f))
#             self.harmony_memory.append((x, f))

#             # Update the best solution
#             if f < self.best_solution[1]:
#                 self.best_solution = (x, f)

#         # Return the best solution
#         return self.best_solution[0]
# ```