import numpy as np
import random
from scipy.optimize import differential_evolution

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.best_individual = None
        self.best_fitness = float('inf')

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
                next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        # Update the best individual and its fitness
        self.best_individual = self.population[np.argmin(self.fitnesses)]
        self.best_fitness = min(self.fitnesses)

        # Refine the strategy
        for _ in range(10):
            new_individual = self.evaluate_fitness(self.best_individual)
            updated_individual = self.f(individual=new_individual, self.logger=self.logger)
            self.population = np.array([updated_individual])
            self.fitnesses = np.array([self.f(updated_individual)])
            self.population_history.append(self.population)
            self.best_individual = updated_individual
            self.best_fitness = self.fitnesses[-1]

        return self.population

    def f(self, individual):
        # Refine the strategy
        x = individual.copy()
        x += np.random.normal(0, 1, size=self.dim)
        return self.evaluate_budget(self.f, x, self.budget)