# class HyperNeighborhood:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population = []
#         self.fitness_values = []
#         self.drift_rate = 0.1  # New line added
#         self.drift_count = 0

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

#             # Apply genetic drift with a probability of self.drift_rate
#             if np.random.rand() < self.drift_rate:
#                 self.drift_count += 1
#                 if self.drift_count % 10 == 0:
#                     # Perform a random replacement of 10% of the population
#                     replacement_indices = np.random.choice(self.population, size=int(self.budget*0.1), replace=False)
#                     self.population[replacement_indices] = np.random.uniform(-5.0, 5.0, self.dim)

#     def evaluate(self):
#         # Evaluate the fitness of each solution in the population
#         self.fitness_values = [func(solution) for solution in self.population]