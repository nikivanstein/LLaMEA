import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.mutation_rate = mutation_rate

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
                if random.random() < self.mutation_rate:
                    mutated_individual[i] += random.uniform(-1, 1)
            return mutated_individual

        def select_parents(population):
            parents = np.random.choice(len(population), size=self.population_size, replace=False)
            return np.array([population[i] for i in parents])

        def crossover(parent1, parent2):
            child = parent1.copy()
            for i in range(self.dim):
                if random.random() < 0.5:
                    child[i] = parent2[i]
            return child

        def mutate_bounding(parent):
            mutated_parent = parent.copy()
            for i in range(self.dim):
                if random.random() < self.mutation_rate:
                    mutated_parent[i] += random.uniform(-1, 1)
            return mutated_parent

        for _ in range(self.budget):
            new_population = np.zeros((self.population_size, self.dim))
            parents = select_parents(self.population)
            for i in range(self.population_size):
                parent1 = parents[i]
                parent2 = parents[(i + 1) % self.population_size]
                child = crossover(parent1, parent2)
                child = mutate_bounding(child)
                fitness = evaluate_budget(eval_func, child, self.budget)
                new_population[i] = child
                self.fitnesses[i] = fitness
                self.population_history.append(new_population[i])

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

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies