import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        population_size = 100
        population = [self.evaluate_fitness(np.random.rand(self.dim)) for _ in range(population_size)]
        while True:
            for _ in range(self.budget):
                new_individuals = self.evaluate_fitness(population)
                best_individual = max(new_individuals)
                if np.random.rand() < 0.05:
                    # Exploration strategy: choose a random individual
                    new_individual = random.choice(population)
                else:
                    # Exploitation strategy: choose the best individual
                    best_individual = max(new_individuals)
                    new_individual = best_individual
                new_individuals = [self.evaluate_fitness(new_individual) for new_individual in new_individuals]
                population = new_individuals
                population = np.delete(population, 0, axis=0)
                self.search_space = np.vstack((self.search_space, new_individual))
                self.search_space = np.delete(self.search_space, 0, axis=0)

    def evaluate_fitness(self, individual):
        return self.func(individual)