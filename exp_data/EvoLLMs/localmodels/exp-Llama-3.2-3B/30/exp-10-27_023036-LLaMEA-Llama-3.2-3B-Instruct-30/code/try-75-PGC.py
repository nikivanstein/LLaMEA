import numpy as np
import random

class PGC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.fitnesses = [self.evaluate_func(func, x) for x in self.population]

    def initialize_population(self):
        population = []
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(x)
        return population

    def evaluate_func(self, func, x):
        return func(x)

    def crossover(self, parent1, parent2):
        child = parent1 + (parent2 - parent1) * random.random()
        return child

    def mutate(self, child):
        idx = random.randint(0, self.dim - 1)
        child[idx] += random.uniform(-1.0, 1.0)
        return child

    def select_parents(self):
        scores = [self.fitnesses[i] for i in range(len(self.fitnesses))]
        scores = np.array(scores)
        parents = []
        for _ in range(len(self.population)):
            idx = np.argmax(scores)
            parents.append(self.population[idx])
            scores[idx] = -np.inf
        return parents

    def optimize(self):
        for _ in range(self.budget):
            parents = self.select_parents()
            new_population = []
            for _ in range(len(self.population)):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
            self.fitnesses = [self.evaluate_func(func, x) for x in self.population]

# Example usage:
def func(x):
    return np.sum(x**2)

pgc = PGC(budget=100, dim=10)
pgc.optimize()