import numpy as np
import random
import copy

class GeneticAlgorithm:
    def __init__(self, budget, dim, mutation_rate, adaptation_rate):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = mutation_rate
        self.adaptation_rate = adaptation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.search_space)
            population.append(copy.deepcopy(individual))
        return population

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individuals = []
            for _ in range(self.population_size):
                new_individual = copy.deepcopy(self.population[-1])
                if np.isnan(new_individual[0]) or np.isinf(new_individual[0]):
                    raise ValueError("Invalid function value")
                if new_individual[0] < -5.0 or new_individual[0] > 5.0:
                    raise ValueError("Function value must be between -5.0 and 5.0")
                if np.isnan(new_individual[1]) or np.isinf(new_individual[1]):
                    raise ValueError("Invalid function value")
                if np.isnan(new_individual[2]) or np.isinf(new_individual[2]):
                    raise ValueError("Invalid function value")
                new_individual[0] = np.clip(new_individual[0], -5.0, 5.0)
                new_individual[1] = np.clip(new_individual[1], 0.0, 1.0)
                new_individual[2] = np.clip(new_individual[2], 0.0, 1.0)
                new_individuals.append(new_individual)
            new_population = []
            for _ in range(self.population_size):
                fitness = self.evaluate_fitness(new_individual)
                if np.isnan(fitness) or np.isinf(fitness):
                    raise ValueError("Invalid fitness value")
                if fitness < 0 or fitness > 1:
                    raise ValueError("Fitness value must be between 0 and 1")
                new_individuals.append(copy.deepcopy(new_individual))
            population = new_population
        return self.evaluate_fitness(population[-1])

    def evaluate_fitness(self, individual):
        func_value = individual[0]
        func_value = func_value * individual[1]
        func_value = func_value * individual[2]
        return func_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm with Adaptation
# Code: 
# ```python
# import numpy as np
# import random
# import copy

# class GeneticAlgorithm:
#     def __init__(self, budget, dim, mutation_rate, adaptation_rate):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = np.linspace(-5.0, 5.0, dim)
#         self.func_evaluations = 0
#         self.population_size = 100
#         self.mutation_rate = mutation_rate
#         self.adaptation_rate = adaptation_rate
#         self.population = self.initialize_population()

#     def initialize_population(self):
#         population = []
#         for _ in range(self.population_size):
#             individual = np.random.uniform(self.search_space)
#             population.append(copy.deepcopy(individual))
#         return population

#     def __call__(self, func):
#         while self.func_evaluations < self.budget:
#             func_evaluations += 1
#             new_individuals = []
#             for _ in range(self.population_size):
#                 new_individual = copy.deepcopy(self.population[-1])
#                 if np.isnan(new_individual[0]) or np.isinf(new_individual[0]):
#                     raise ValueError("Invalid function value")
#                 if new_individual[0] < -5.0 or new_individual[0] > 5.0:
#                     raise ValueError("Function value must be between -5.0 and 5.0")
#                 if np.isnan(new_individual[1]) or np.isinf(new_individual[1]):
#                     raise ValueError("Invalid function value")
#                 if np.isnan(new_individual[2]) or np.isinf(new_individual[2]):
#                     raise ValueError("Invalid function value")
#                 new_individual[0] = np.clip(new_individual[0], -5.0, 5.0)
#                 new_individual[1] = np.clip(new_individual[1], 0.0, 1.0)
#                 new_individual[2] = np.clip(new_individual[2], 0.0, 1.0)
#                 new_individuals.append(new_individual)
#             new_population = []
#             for _ in range(self.population_size):
#                 fitness = self.evaluate_fitness(new_individual)
#                 if np.isnan(fitness) or np.isinf(fitness):
#                     raise ValueError("Invalid fitness value")
#                 if fitness < 0 or fitness > 1:
#                     raise ValueError("Fitness value must be between 0 and 1")
#                 new_individuals.append(copy.deepcopy(new_individual))
#             population = new_population
#         return self.evaluate_fitness(population[-1])

# HEBBO: Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm with Adaptation
# Code: 
# ```python
# import numpy as np

# class HEBBO:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = np.linspace(-5.0, 5.0, dim)
#         self.func_evaluations = 0

#     def __call__(self, func):
#         while self.func_evaluations < self.budget:
#             func_evaluations += 1
#             func_value = func(self.search_space)
#             if np.isnan(func_value) or np.isinf(func_value):
#                 raise ValueError("Invalid function value")
#             if func_value < 0 or func_value > 1:
#                 raise ValueError("Function value must be between 0 and 1")
#             self.search_space = np.linspace(-5.0, 5.0, self.dim)
#         return func_value

# An exception occured: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 52, in evaluateBBOB
#     algorithm(problem)
#   File "<string>", line 12, in __call__
#     UnboundLocalError: cannot access local variable 'func_evaluations' where it is not associated with a value