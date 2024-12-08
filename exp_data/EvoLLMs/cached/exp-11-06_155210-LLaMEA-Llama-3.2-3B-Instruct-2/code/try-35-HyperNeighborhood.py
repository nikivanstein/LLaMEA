# class HyperNeighborhood:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population = []
#         self.fitness_values = []
#         self.crossover_rate = 0.2  # Enhanced selection
#         self.mutation_rate = 0.1  # Adaptive mutation

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

#             # Apply crossover and mutation
#             if np.random.rand() < self.crossover_rate:
#                 parent1, parent2 = np.random.choice(self.population, size=2, replace=False)
#                 child = (parent1 + parent2) / 2
#                 self.population.remove(parent1)
#                 self.population.remove(parent2)
#                 self.population.append(child)

#             if np.random.rand() < self.mutation_rate:
#                 index = np.random.randint(0, self.dim)
#                 self.population[index] += np.random.uniform(-0.2, 0.2)