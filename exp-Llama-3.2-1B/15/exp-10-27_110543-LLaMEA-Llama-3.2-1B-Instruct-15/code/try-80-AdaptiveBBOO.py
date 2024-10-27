import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.deltas = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def mutate(x):
            if random.random() < 0.1:
                return x + np.random.normal(0, 1, size=self.dim)
            return x

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
                new_population = np.zeros((self.population_size, self.dim))
                for i in range(self.population_size):
                    x = mutate(self.population[i])
                    fitness = evaluate_budget(eval_func, x, self.budget)
                    new_population[i] = x
                    self.population[i] = x
                    self.deltas[i] = fitness - self.fitnesses[i]

                # Replace the old population with the new one
                self.population = new_population
                self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Select the fittest individuals with probability 0.15
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            next_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                x = self.population[i] + 0.1 * self.deltas[i]
                fitness = evaluate_budget(eval_func, x, self.budget)
                next_population[i] = x
                self.population[i] = x

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies