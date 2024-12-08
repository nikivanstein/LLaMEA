import numpy as np
import random

class AdaptiveBBOOPEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.evolutionary_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        bounds = bounds(individual)
        if random.random() < 0.2:
            new_dim = random.randint(1, self.dim)
            new_individual = individual.copy()
            new_individual[new_dim] = random.uniform(bounds[0], bounds[1])
            self.population[i] = new_individual
            self.evolutionary_history.append((individual, new_individual))

    def evaluate_fitness(self, individual):
        updated_individual = self.f(individual, self.logger)
        if len(self.evolutionary_history) < 10:
            self.evolutionary_history.append((individual, updated_individual))
        return updated_individual

    def update_population(self, new_individual):
        for i in range(self.population_size):
            if new_individual not in self.population[i]:
                self.population[i] = new_individual
                self.fitnesses[i] = new_individual

    def __str__(self):
        return f"Population Size: {self.population_size}\nFitnesses: {self.fitnesses}\nEvolutionary History: {self.evolutionary_history}"

# BBOOPEvolutionaryStrategy
# Code: 