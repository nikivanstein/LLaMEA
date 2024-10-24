# import numpy as np
# import random

# class EvolutionaryGradient:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.x = np.random.uniform(-5.0, 5.0, size=dim)
#         self.f_best = np.inf
#         self.x_best = None

#     def __call__(self, func):
#         for _ in range(self.budget):
#             # Compute gradient of the objective function
#             gradient = np.zeros(self.dim)
#             h = 1e-1
#             for i in range(self.dim):
#                 gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

#             # Update the current solution using evolutionary strategy
#             self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)

#             # Update the best solution
#             f = func(self.x)
#             if f < self.f_best:
#                 self.f_best = f
#                 self.x_best = self.x.copy()

#             # Add gradient information to the evolutionary strategy
#             self.x += 0.1 * gradient

#             # Check for convergence
#             if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
#                 print("Converged after {} iterations".format(_))

#     def refine(self):
#         # Probability to change the individual lines
#         p = 0.025

#         # Change the individual lines with probability p
#         if random.random() < p:
#             for i in range(self.dim):
#                 self.x[i] += np.random.uniform(-0.1, 0.1)

#         # Check for convergence
#         f = self.f(self.x)
#         if f < self.f_best:
#             self.f_best = f
#             self.x_best = self.x.copy()

# def evaluate_fitness(individual):
#     # Evaluate the fitness of the individual
#     return individual.f_best

# def mutate(individual):
#     # Mutate the individual
#     return individual

# def crossover(parent1, parent2):
#     # Perform crossover between two parents
#     child = np.zeros_like(parent1)
#     for i in range(len(parent1)):
#         if random.random() < 0.5:
#             child[i] = parent1[i]
#         else:
#             child[i] = parent2[i]
#     return child

# def evaluateBBOB(func, population_size=50, max_evals=1000, dim=10):
#     # Evaluate the fitness of the population
#     population = []
#     for _ in range(population_size):
#         individual = EvolutionaryGradient(max_evals, dim)
#         individual("func")
#         population.append(individual)

#     # Select the fittest individuals
#     population = sorted(population, key=lambda x: x.f_best)

#     # Evaluate the fitness of the selected individuals
#     fitness = []
#     for individual in population:
#         fitness.append(evaluate_fitness(individual))

#     return population, fitness

# def initialize_single(individual):
#     # Initialize the individual
#     individual.refine()
#     return individual

# def mutate_single(individual):
#     # Mutate the individual
#     individual.refine()
#     return individual

# def crossover_single(parent1, parent2):
#     # Perform crossover between two parents
#     child = EvolutionaryGradient(max_evals, dim)
#     child("func")
#     return child

# def evaluateBBOB_single(func, population_size=50, max_evals=1000, dim=10):
#     # Evaluate the fitness of the population
#     population = []
#     for _ in range(population_size):
#         individual = EvolutionaryGradient(max_evals, dim)
#         individual("func")
#         population.append(individual)

#     # Select the fittest individuals
#     population = sorted(population, key=lambda x: x.f_best)

#     # Evaluate the fitness of the selected individuals
#     fitness = []
#     for individual in population:
#         fitness.append(evaluate_fitness(individual))

#     return population, fitness