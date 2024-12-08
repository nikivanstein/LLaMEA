import numpy as np
import random

class AdaptiveHEBO:
    def __init__(self, budget, dim, alpha=0.1, beta=0.05):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 10
        self.diversity = 0.1
        self.alpha = alpha
        self.beta = beta
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            # Select new individuals with adaptive population size and diversity
            new_individuals = []
            for _ in range(self.population_size):
                if np.random.rand() < self.alpha:
                    # Select new individual with random mutation
                    new_individual = func(self.search_space)
                else:
                    # Select new individual with diversity mutation
                    new_individual = self.select_new_individual(func, self.search_space)
                new_individuals.append(new_individual)
            # Evaluate new individuals and select the best one
            new_individuals = self.evaluate_new_individuals(new_individuals, func)
            # Select the best individual based on diversity and fitness
            new_individual = self.select_best_individual(new_individuals, func)
            # Replace the old population with the new one
            self.population = new_individuals
            self.func_evaluations = 0
        return self.population[0]

    def select_new_individual(self, func, search_space):
        # Select new individual with random mutation
        new_individual = func(np.random.rand(self.dim))
        # Select new individual with diversity mutation
        while True:
            new_individual = self.select_diversityMutation(func, new_individual, search_space)
            if np.random.rand() < self.beta:
                break
        return new_individual

    def select_diversityMutation(self, func, new_individual, search_space):
        # Select new individual with diversity mutation
        new_individual = np.random.choice(search_space, size=self.dim, replace=True)
        for i in range(self.dim):
            new_individual[i] = func(new_individual[i])
        return new_individual

    def evaluate_new_individuals(self, new_individuals, func):
        # Evaluate new individuals and return the best one
        best_individual = None
        best_fitness = -np.inf
        for individual in new_individuals:
            fitness = func(individual)
            if fitness > best_fitness:
                best_individual = individual
                best_fitness = fitness
        return [best_individual]

    def select_best_individual(self, new_individuals, func):
        # Select the best individual based on diversity and fitness
        best_individual = None
        best_fitness = -np.inf
        for individual in new_individuals:
            fitness = func(individual)
            if fitness > best_fitness:
                best_individual = individual
                best_fitness = fitness
        return best_individual

# Description: Heuristic Optimization using Adaptive Population Size and Diversity
# Code: 