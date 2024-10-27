import random
import numpy as np
import operator

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)
        self.population_size = 100
        self.population = np.random.choice(self.search_space, size=(self.population_size, self.dim), p=self.get_probability())

    def get_probability(self, individual):
        # Normalize the probability distribution over the search space
        probabilities = np.array([self.get_fitness(individual, self.func) for _ in range(self.population_size)])
        probabilities /= np.sum(probabilities)
        return probabilities

    def get_fitness(self, individual, func):
        # Evaluate the fitness of an individual using the provided function
        return func(individual)

    def __call__(self, func):
        # Initialize the population using a random selection of individuals
        while True:
            self.population = np.random.choice(self.search_space, size=(self.population_size, self.dim), p=self.get_probability())
            for _ in range(self.budget):
                new_individual = self.evaluate_fitness(self.population)
                if np.linalg.norm(func(new_individual)) < self.budget / 2:
                    self.population = new_individual
                    break
            self.population = np.vstack((self.population, self.population[-1]))
            self.population = np.delete(self.population, 0, axis=0)
            self.search_space = np.vstack((self.search_space, self.population[-1]))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual using the provided function
        return self.get_fitness(individual, self.func)

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np
# import operator
#
# class BBOBOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
#         self.func = lambda x: np.sum(x)
#         self.population_size = 100
#         self.population = np.random.choice(self.search_space, size=(self.population_size, self.dim), p=self.get_probability())
#
#     def get_probability(self, individual):
#         # Normalize the probability distribution over the search space
#         probabilities = np.array([self.get_fitness(individual, self.func) for _ in range(self.population_size)])
#         probabilities /= np.sum(probabilities)
#         return probabilities
#
#     def get_fitness(self, individual, func):
#         # Evaluate the fitness of an individual using the provided function
#         return func(individual)
#
#     def __call__(self, func):
#         # Initialize the population using a random selection of individuals
#         while True:
#             self.population = np.random.choice(self.search_space, size=(self.population_size, self.dim), p=self.get_probability())
#             for _ in range(self.budget):
#                 new_individual = self.evaluate_fitness(self.population)
#                 if np.linalg.norm(func(new_individual)) < self.budget / 2:
#                     self.population = new_individual
#                     break
#             self.population = np.vstack((self.population, self.population[-1]))
#             self.search_space = np.vstack((self.search_space, self.population[-1]))
#             self.search_space = np.delete(self.search_space, 0, axis=0)
#
# BBOBOptimizer(100, 2).__call__(np.random.rand(2))