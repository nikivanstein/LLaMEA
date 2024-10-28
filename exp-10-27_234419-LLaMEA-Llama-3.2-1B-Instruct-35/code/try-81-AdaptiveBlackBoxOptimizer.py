import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.population = []
        self.population_history = []
        self.fitness_history = []
        self.population_size = 100
        self.population_count = 0

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

        self.population.append(func(self.sample_indices))
        self.population_history.append(self.population)
        self.fitness_history.append([np.abs(best_func - func(self.sample_indices)) for best_func in self.population])

        if len(self.population) >= self.population_size:
            self.population = self.population[:self.population_size]
            self.population_count = 0

        self.population_count += 1

        return func(self.sample_indices)

    def update(self, new_individual):
        if len(self.population) >= self.population_size:
            self.population = self.population[:self.population_size]
            self.population_count = 0

        # Refine the strategy using the probability 0.35
        if random.random() < 0.35:
            new_individual = self.evaluate_fitness(new_individual)
        else:
            new_individual = self.evaluate_fitness(np.random.choice(self.population, size=len(self.population), replace=True))

        return new_individual

    def evaluate_fitness(self, individual):
        return np.abs(individual - self.func(individual))

# One-line description with the main idea
# Adaptive Black Box Optimization using Genetic Algorithm
# 
# This algorithm uses a genetic algorithm to optimize a black box function by evolving a population of individuals with a probability of 0.35.
# The population size is dynamically adjusted based on the fitness history to ensure that the best individuals are selected for reproduction.
# 
# Code: 