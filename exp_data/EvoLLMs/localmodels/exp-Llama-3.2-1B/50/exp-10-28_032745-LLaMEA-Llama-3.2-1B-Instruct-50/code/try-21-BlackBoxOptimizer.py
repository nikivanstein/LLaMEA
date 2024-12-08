import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        population = self.initialize_population(iterations)
        for _ in range(iterations):
            if _ >= self.budget:
                break
            new_population = self.select_parents(population)
            new_population = self.evaluate_new_population(new_population, func, self.search_space)
            new_population = self.reproduce(new_population)
            new_population = self.evaluate_new_population(new_population, func, self.search_space)
            population = new_population
        return population[0], population[1]

    def initialize_population(self, iterations):
        population = []
        for _ in range(iterations):
            individual = copy.deepcopy(self.func(np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)])))
            population.append(individual)
        return population

    def select_parents(self, population):
        parents = []
        for _ in range(int(self.budget * 0.5)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            while parent1 == parent2:
                parent2 = random.choice(population)
            parents.append((parent1, parent2))
        return parents

    def evaluate_new_population(self, new_population, func, search_space):
        new_population = []
        for individual in new_population:
            updated_individual = copy.deepcopy(individual)
            for i in range(self.dim):
                new_value = func(updated_individual)
                if new_value < updated_individual[i] + random.uniform(-0.01, 0.01):
                    updated_individual[i] += random.uniform(-0.01, 0.01)
            new_population.append(updated_individual)
        return new_population

    def reproduce(self, population):
        children = []
        for _ in range(int(len(population) * 0.75)):
            parent1, parent2 = random.sample(population, 2)
            child = (parent1 + parent2) / 2
            children.append(child)
        return children

    def evaluate_fitness(self, individual, logger):
        updated_individual = copy.deepcopy(individual)
        for i in range(self.dim):
            new_value = self.func(updated_individual)
            if new_value < updated_individual[i] + random.uniform(-0.01, 0.01):
                updated_individual[i] += random.uniform(-0.01, 0.01)
        logger.update_fitness(individual, updated_individual)
        return updated_individual

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 