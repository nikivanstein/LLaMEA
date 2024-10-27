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

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def mutate(individual):
            mutated_individual = individual.copy()
            for i in range(self.dim):
                if random.random() < 0.15:
                    mutated_individual[i] += random.uniform(-1, 1)
            return mutated_individual

        def crossover(parent1, parent2):
            child = parent1.copy()
            for i in range(self.dim):
                if random.random() < 0.15:
                    child[i] = parent2[i]
            return child

        def select_parents(fittest_individuals, num_parents):
            return fittest_individuals[np.argsort(fittest_individuals, axis=0)[:num_parents]]

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = evaluate_budget(eval_func, self.population[i], self.budget)
                self.fitnesses[i] = fitness
                self.population_history.append(self.population[i])

            # Select the fittest individuals
            self.population = select_parents(self.fitnesses, self.population_size)
            self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

            # Evolve the population
            for _ in range(100):
                next_population = np.zeros((self.population_size, self.dim))
                for i in range(self.population_size):
                    parent1 = self.population[i]
                    parent2 = self.population[i]
                    if i < self.population_size // 2:
                        parent1 = self.select_parents([parent1, parent2], self.population_size // 2)[0]
                    else:
                        parent2 = self.select_parents([parent1, parent2], self.population_size // 2)[0]
                    child = crossover(parent1, parent2)
                    next_population[i] = mutate(child)
                    self.population[i] = next_population[i]

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Genetic Algorithm