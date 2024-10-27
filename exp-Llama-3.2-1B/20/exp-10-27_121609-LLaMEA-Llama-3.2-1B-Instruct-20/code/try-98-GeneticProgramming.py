import numpy as np
import random
import operator

class GeneticProgramming:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_dict = {}
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        new_individual = individual.copy()
        if random.random() < 0.2:
            new_individual[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
        return new_individual

    def evaluate_fitness(self, individual):
        updated_individual = individual
        while True:
            if updated_individual in self.population_dict:
                updated_individual = self.population_dict[updated_individual]
                break
            fitness = self.__call__(updated_individual)
            if fitness < self.fitnesses[updated_individual, updated_individual] + 1e-6:
                self.fitnesses[updated_individual, updated_individual] = fitness
                self.population_dict[updated_individual] = updated_individual
            else:
                break
        return updated_individual

    def __str__(self):
        return "GeneticProgramming: Optimizes black box function using genetic programming"

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# 
# This algorithm uses a genetic programming approach to optimize black box functions
# by evolving a population of candidate solutions through a process of mutation and selection
# 
# The algorithm is based on the idea of evolving a population of candidate solutions
# through a process of mutation and selection, where each individual in the population
# is mutated to introduce new variations and then evaluated to determine its fitness
# 
# The mutation rate is controlled by a probability (0.2) that determines the likelihood
# of a mutation occurring in an individual