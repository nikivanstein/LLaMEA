import numpy as np
import random

class DynamicEvoOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.iterations = 0
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def adaptive_line_search(x, alpha):
            return x + alpha * (objective(x) - objective(x.min()))

        def new_individual():
            return self.evaluate_fitness()

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                return individual
            else:
                return new_individual()

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    new_individual = evaluate_fitness(x)
                    self.population[i] = new_individual
                    self.population_history.append(new_individual)

            if random.random() < 0.2:
                alpha = random.uniform(0.1, 1.0)
                x = adaptive_line_search(x, alpha)
                self.population[i] = x

        return self.fitnesses

    def evaluate_fitness(self):
        new_individual = self.population[0]
        for _ in range(self.budget):
            new_individual = self.evaluate_fitness(new_individual)
        return new_individual

# Description: Dynamic Evolutionary Optimization with Adaptive Line Search
# Code: 
# ```python
# DynamicEvoOpt(budget=100, dim=10)
# ```