import numpy as np
import random

class MetaHeuristic:
    def __init__(self, budget, dim, mutation_rate=0.2, crossover_rate=0.8):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if random.random() < self.mutation_rate:
                idx = random.randint(0, self.dim - 1)
                individual[idx] += random.uniform(-1, 1)
                if individual[idx] < -5.0:
                    individual[idx] = -5.0
                elif individual[idx] > 5.0:
                    individual[idx] = 5.0

        def crossover(parent1, parent2):
            if random.random() < self.crossover_rate:
                idx = random.randint(0, self.dim - 1)
                child = parent1[:idx] + parent2[idx:]
                return child

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                return individual
            else:
                return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.evaluate_fitness(self.population[i])
                if x is None:
                    x = evaluate_fitness(x)
                mutate(x)
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using mutation and crossover strategies