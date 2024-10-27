import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        population = [self.evaluate_fitness(x) for x in self.search_space]
        while True:
            population = [self.evaluate_fitness(x) for x in population]
            if np.mean(population) <= self.budget / 2:
                break
            new_individual = self.select_individual(population)
            if np.random.rand() < 0.05:
                new_individual = self.refine_strategy(new_individual)
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def select_individual(self, population):
        selection_probabilities = np.array([population[i] / np.sum(population) for i in range(len(population))])
        return np.random.choice(len(population), p=selection_probabilities)

    def refine_strategy(self, individual):
        # Simple strategy: select the individual that has a fitness score closest to the mean
        mean_fitness = np.mean([self.evaluate_fitness(individual) for individual in self.search_space])
        return self.search_space[np.argmin([self.evaluate_fitness(individual) for individual in self.search_space])]

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 