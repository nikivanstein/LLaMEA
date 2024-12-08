import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.selection_rate = 0.5

    def __call__(self, func):
        # Initialize population
        population = self.initialize_population()

        # Evaluate population
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

        # Select parents
        parents = self.select_parents(population, fitnesses)

        # Crossover
        offspring = self.crossover(parents)

        # Mutate
        offspring = self.mutate(offspring)

        # Evaluate new population
        new_fitnesses = [self.evaluate_fitness(individual, func) for individual in offspring]

        # Select new parents
        new_parents = self.select_parents(new_fitnesses, new_fitnesses)

        # Replace old population
        population = new_parents

        # Check for convergence
        if self.check_convergence(population, fitnesses):
            return population[0]

        return self.__call__(func)

    def initialize_population(self):
        return [np.random.uniform(self.search_space) for _ in range(self.population_size)]

    def evaluate_fitness(self, individual, func):
        while True:
            func_value = func(individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if func_value < 0.5:
                return 0
            elif func_value > 0.8:
                return 1
            else:
                return 0.5

    def select_parents(self, fitnesses, new_fitnesses):
        parents = []
        for _ in range(self.population_size):
            fitness = fitnesses[_]
            new_fitness = new_fitnesses[_]
            if random.random() < self.selection_rate:
                parents.append(individual = self.evaluate_fitness(individual, func = fitness))
        return parents

    def crossover(self, parents):
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual + random.uniform(-1, 1)
            if np.isnan(mutated_individual) or np.isinf(mutated_individual):
                mutated_individual = individual
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def check_convergence(self, population, fitnesses):
        convergence = True
        for fitness in fitnesses:
            if fitness < 0 or fitness > 1:
                convergence = False
                break
        return convergence