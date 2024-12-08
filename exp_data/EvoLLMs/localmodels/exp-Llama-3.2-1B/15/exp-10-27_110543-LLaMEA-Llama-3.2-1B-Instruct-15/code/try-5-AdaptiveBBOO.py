import numpy as np
from scipy.optimize import minimize
from collections import deque

class AdaptiveBBOO:
    def __init__(self, budget, dim, adaptive_bounds):
        self.budget = budget
        self.dim = dim
        self.adaptive_bounds = adaptive_bounds
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.mutation_rate = 0.01
        self.mutation_bounds = 0.1

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = evaluate_budget(eval_func, self.population[i], self.budget)
                self.fitnesses[i] = fitness
                self.population_history.append(self.population[i])

        # Select the fittest individuals
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            next_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                new_individual = self.evaluate_fitness(new_individual)
                new_individual = self.adaptive_bounds[0] + np.random.uniform(-self.adaptive_bounds[1], self.adaptive_bounds[0]) * np.random.normal(0, 1, size=self.dim)
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 1, size=self.dim)
                    new_individual = new_individual + mutation
                    new_individual = self.adaptive_bounds[0] + np.random.uniform(-self.adaptive_bounds[1], self.adaptive_bounds[0]) * np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, new_individual, self.budget)
                next_population[i] = new_individual
                self.population[i] = new_population[i]
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        return self.population

    def evaluate_fitness(self, individual):
        return evaluate_budget(eval_func, individual, self.budget)

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies with adaptive bounds and mutation