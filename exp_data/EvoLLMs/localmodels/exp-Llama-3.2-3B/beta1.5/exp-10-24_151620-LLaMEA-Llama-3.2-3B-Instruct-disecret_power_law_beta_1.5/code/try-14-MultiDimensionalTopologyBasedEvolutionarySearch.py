import numpy as np
import random

class MultiDimensionalTopologyBasedEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.topology = self.initialize_topology()
        self.mutation_rate = 0.07547169811320754

    def initialize_topology(self):
        topology = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            topology[i, i] = 1
        for i in range(1, self.dim):
            for j in range(i):
                if random.random() < 0.5:
                    topology[i, j] = 1
                    topology[j, i] = 1
        return topology

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, 1))
        for _ in range(self.budget):
            fitness = np.array([func(x) for x in population.flatten()])
            indices = np.argsort(fitness)
            population = population[indices]
            new_population = population[:self.dim//2]
            new_population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim//2, 1))
            new_population = np.c_[new_population, population[self.dim//2:]]
            population = np.concatenate((new_population, population[:self.dim//2]))
            self.topology = self.update_topology(population)
            population = population.flatten()
            self.mutate(population, self.mutation_rate)
        return np.min(fitness)

    def update_topology(self, population):
        new_topology = self.initialize_topology()
        for i in range(self.dim):
            for j in range(self.dim):
                if population[i] < population[j]:
                    new_topology[i, j] = 1
                elif population[i] > population[j]:
                    new_topology[j, i] = 1
        return new_topology

    def mutate(self, population, mutation_rate):
        mutated_population = population.copy()
        for i in range(self.dim):
            if random.random() < mutation_rate:
                mutated_population[i] += np.random.uniform(-1, 1)
                mutated_population[i] = np.clip(mutated_population[i], self.search_space[0], self.search_space[1])
        mutated_population = np.sort(mutated_population)
        for i in range(self.dim):
            if mutated_population[i] < population[i]:
                mutated_population[i] = population[i]
            elif mutated_population[i] > population[i]:
                mutated_population[i] = population[i]
        return mutated_population

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
opt = MultiDimensionalTopologyBasedEvolutionarySearch(budget, dim)
print(opt(func))