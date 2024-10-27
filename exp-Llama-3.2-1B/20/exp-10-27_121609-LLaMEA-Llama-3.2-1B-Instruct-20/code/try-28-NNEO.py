import numpy as np
from collections import deque
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = deque(maxlen=10)

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if random.random() < 0.2:
                return x + np.random.uniform(-0.5, 0.5)
            else:
                return x

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual[0], individual[0]] + 1e-6:
                self.fitnesses[individual[0], individual[0]] = fitness
                return individual
            else:
                return individual

        def select_parent(population):
            return random.choice(population)

        def crossover(parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                if random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(i)
                if individual not in self.population_history:
                    self.population_history.append(individual)
                if len(self.population_history) > self.population_size:
                    self.population_history.popleft()

                new_individual = crossover(self.population[i], individual)
                new_individual = mutate(new_individual)
                new_individual = evaluate_fitness(new_individual)

                if new_individual not in self.population_history:
                    self.population_history.append(new_individual)

                if new_individual in self.population:
                    self.population[i] = new_individual

                if new_individual in self.fitnesses:
                    self.fitnesses[new_individual[0], new_individual[0]] = new_individual

        return self.fitnesses

# Description: NNEO uses a combination of mutation, crossover, and selection to optimize black box functions.
# Code:
# ```python
# import numpy as np
# import random
# import math

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = deque(maxlen=10)

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if random.random() < 0.2:
                return x + np.random.uniform(-0.5, 0.5)
            else:
                return x

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual[0], individual[0]] + 1e-6:
                self.fitnesses[individual[0], individual[0]] = fitness
                return individual
            else:
                return individual

        def select_parent(population):
            return random.choice(population)

        def crossover(parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                if random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(i)
                if individual not in self.population_history:
                    self.population_history.append(individual)
                if len(self.population_history) > self.population_size:
                    self.population_history.popleft()

                new_individual = crossover(self.population[i], individual)
                new_individual = mutate(new_individual)
                new_individual = evaluate_fitness(new_individual)

                if new_individual not in self.population_history:
                    self.population_history.append(new_individual)

                if new_individual in self.population:
                    self.population[i] = new_individual

                if new_individual in self.fitnesses:
                    self.fitnesses[new_individual[0], new_individual[0]] = new_individual

        return self.fitnesses