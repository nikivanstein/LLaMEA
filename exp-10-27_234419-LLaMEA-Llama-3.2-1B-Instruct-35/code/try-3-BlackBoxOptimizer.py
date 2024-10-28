import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.population_size = 100

    def __call__(self, func):
        population = [BlackBoxOptimizer(self.budget, dim) for _ in range(self.population_size)]

        while True:
            fitness_values = []
            for individual in population:
                fitness = individual(func)
                fitness_values.append(fitness)

            fitness_values = np.array(fitness_values)
            fitness_values = np.sort(fitness_values)
            fitness_values = fitness_values[-self.population_size:]

            selected_indices = np.random.choice(self.population_size, size=self.population_size, replace=False)
            selected_individuals = [population[i] for i in selected_indices]

            for individual in selected_individuals:
                individual.local_search = True

            for _ in range(100):
                if self.local_search:
                    best_individual = None
                    best_fitness = -1e10
                    for individual in selected_individuals:
                        fitness = individual.func(self.sample_indices)
                        if fitness > best_fitness:
                            best_individual = individual
                            best_fitness = fitness
                    selected_individuals.remove(best_individual)
                    selected_individuals.append(best_individual)

                if len(selected_individuals) == 0:
                    break

            selected_individuals = [individual for individual in selected_individuals if individual.local_search]

            new_population = []
            for _ in range(self.population_size):
                if self.sample_indices is None:
                    new_individual = BlackBoxOptimizer(self.budget, dim)
                else:
                    new_individual = BlackBoxOptimizer(self.budget, dim)
                    new_individual.search_space = np.random.uniform(-5.0, 5.0, self.dim)
                    new_individual.sample_size = 1
                    new_individual.sample_indices = None
                    new_individual.local_search = False

                while True:
                    fitness = new_individual.func(new_individual.sample_indices)
                    if fitness > new_individual.sample_indices is None:
                        break
                    new_individual.sample_indices = np.random.choice(new_individual.search_space, size=new_individual.sample_size, replace=False)
                    new_individual.sample_indices = new_individual.sample_indices[:new_individual.sample_size]
                    new_individual.local_search = False

                new_population.append(new_individual)

            population = new_population

            if np.abs(np.mean([individual.func(individual.sample_indices) - func(individual.sample_indices) for individual in population])) < 1e-6:
                break

        return population[0]

# Description: Adaptive Black Box Optimization using Genetic Algorithm with Adaptive Sampling and Local Search
# Code: 