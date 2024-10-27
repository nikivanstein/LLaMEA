import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim, dim_tuning):
        self.budget = budget
        self.dim = dim
        self.dim_tuning = dim_tuning
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, dim))
        self.fitnesses = np.zeros((self.population_size, dim))
        self.population_history = []
        self.dim_tuning_history = []

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
                new_individual = self.evaluate_fitness(new_individual=self.population[i], dim_tuning=self.dim_tuning)
                next_population[i] = new_individual + np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        return self.population

    def evaluate_fitness(self, dim_tuning, individual, dim):
        new_individual = individual + np.random.normal(0, dim, size=dim)
        fitness = evaluate_budget(eval_func, new_individual, self.budget)
        return new_individual, fitness