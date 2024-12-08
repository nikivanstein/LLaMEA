import numpy as np
import random
import time

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10
        self.refine_rate = 0.25

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.budget):
            fitness = np.array([func(x) for x in population])
            fittest = population[np.argsort(fitness)]
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(fittest, 2)
                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
                if random.random() < self.crossover_rate:
                    child = self.crossover(child, parent1, parent2)
                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)
                offspring.append(child)
            population = np.array(offspring)
            if _ % int(self.budget * self.refine_rate) == 0:
                self.refine(population, func)
        return population[np.argmin(fitness)]

    def crossover(self, child, parent1, parent2):
        child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
        return child

    def refine(self, population, func):
        for i in range(self.population_size):
            parent1, parent2 = random.sample(population, 2)
            child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
            child += np.random.uniform(-0.1, 0.1, self.dim)
            fitness1 = func(parent1)
            fitness2 = func(parent2)
            fitness3 = func(child)
            if fitness1 < fitness2 and fitness1 < fitness3:
                population[i] = child
            elif fitness2 < fitness1 and fitness2 < fitness3:
                population[i] = child
            elif fitness3 < fitness1 and fitness3 < fitness2:
                population[i] = child

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)