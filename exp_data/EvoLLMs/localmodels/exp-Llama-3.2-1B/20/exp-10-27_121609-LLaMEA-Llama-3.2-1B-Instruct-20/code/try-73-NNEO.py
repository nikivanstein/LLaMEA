# Description: Novel Hybrid Genetic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

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
        new_individual = individual.copy()
        if random.random() < 0.2:  # 20% chance of mutation
            new_individual[random.randint(0, self.dim-1)] += random.uniform(-0.1, 0.1)
        return new_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        if random.random() < 0.5:  # 50% chance of crossover
            crossover_point = random.randint(0, self.dim-1)
            child[crossover_point] = parent2[crossover_point]
        return child

    def evaluate_fitness(self, individual):
        updated_individual = self.evaluate_fitness(individual)
        if len(updated_individual) == 0:
            return updated_individual
        return updated_individual

    def generate_initial_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

class NNEOHybrid(NNEO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.generate_initial_population()[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        for _ in range(10):  # 10% chance of refinement
            for i in range(self.population_size):
                x = self.population[i]
                new_individual = self.mutate(x)
                new_fitness = objective(new_individual)
                if new_fitness < self.fitnesses[i, new_individual] + 1e-6:
                    self.fitnesses[i, new_individual] = new_fitness
                    self.population[i] = new_individual

        return self.fitnesses

nneo = NNEOHybrid(budget=1000, dim=20)
nneo()