import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim, mutation_rate, selection_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.mutation_count = 0

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
                if random.random() < self.selection_rate:
                    next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                else:
                    next_population[i] = self.population[i] + np.random.normal(0, 0.1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

            # Mutate the population
            for i in range(self.population_size):
                if random.random() < self.mutation_rate:
                    mutation_index = random.randint(0, self.dim - 1)
                    mutation_value = random.uniform(-1, 1)
                    self.population[i][mutation_index] += mutation_value

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies with Dynamic Dimensionality