import numpy as np
import random
from collections import deque

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.population_history_size = 0

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def mutate(individual):
            mutated_individual = individual.copy()
            for _ in range(random.randint(1, self.dim)):
                mutated_individual[random.randint(0, self.dim-1)] += random.uniform(-5.0, 5.0)
            return mutated_individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = evaluate_budget(eval_func, self.population[i], self.budget)
                self.fitnesses[i] = fitness
                self.population_history.append(self.population[i])
                if self.population_history_size < self.budget:
                    self.population_history_size += 1

            # Select the fittest individuals
            self.population = self.population[np.argsort(self.fitnesses, axis=0)]
            self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

            # Evolve the population
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                new_individual = mutate(self.population[i])
                fitness = evaluate_budget(eval_func, new_individual, self.budget)
                new_population[i] = new_individual
                self.population[i] = new_individual
                self.population_history.append(new_population[i])
                if self.population_history_size >= self.budget:
                    break

            # Replace old population with new one
            self.population = self.population[np.argsort(self.fitnesses, axis=0)]
            self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

            # Evolve the population
            for _ in range(100):
                next_population = np.zeros((self.population_size, self.dim))
                for i in range(self.population_size):
                    next_individual = mutate(self.population[i])
                    fitness = evaluate_budget(eval_func, next_individual, self.budget)
                    next_population[i] = next_individual
                    self.population[i] = next_individual
                    self.population_history.append(next_population[i])
                    if self.population_history_size >= self.budget:
                        break

            # Replace old population with new one
            self.population = self.population[np.argsort(self.fitnesses, axis=0)]
            self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies