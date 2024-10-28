import numpy as np
import random
from scipy.optimize import minimize

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_scores = np.zeros(self.population_size)

    def __call__(self, func):
        for _ in range(self.budget):
            func(self.population)
        return self.fitness_scores

    def selection(self):
        self.fitness_scores = self.__call__(self)
        self.fitness_scores = np.sort(self.fitness_scores)
        self.fitness_scores = self.fitness_scores[:self.population_size]
        self.fitness_scores /= self.fitness_scores.sum()
        self.population = self.fitness_scores[np.argsort(self.fitness_scores)]
        return self.population

    def crossover(self):
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for i in range(self.population_size):
            parent1, parent2 = self.population[i], self.population[i + 1]
            if random.random() < 0.5:
                self.population[i] = np.mean([parent1, parent2])
            else:
                self.population[i] = np.median([parent1, parent2])
        return self.population

    def mutation(self):
        for i in range(self.population_size):
            if random.random() < 0.1:
                self.population[i] += random.uniform(-1.0, 1.0)
        return self.population

    def bounds(self):
        return np.random.uniform(-5.0, 5.0, (self.dim,))

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization
# Refine the strategy by iteratively selecting the best individual, 
# crossovering between the best and worst individuals, and introducing random mutations