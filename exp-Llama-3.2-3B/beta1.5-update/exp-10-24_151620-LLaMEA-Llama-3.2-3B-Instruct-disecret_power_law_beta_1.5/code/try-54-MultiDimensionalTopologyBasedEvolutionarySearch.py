import numpy as np
import random

class MultiDimensionalTopologyBasedEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.topology = self.initialize_topology()
        self.adaptive_neighborhood = self.initialize_adaptive_neighborhood()

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

    def initialize_adaptive_neighborhood(self):
        neighborhood = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if i!= j:
                    neighborhood[i, j] = 1
        return neighborhood

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

            # Update topology with adaptive neighborhood
            for i in range(self.dim):
                for j in range(self.dim):
                    if population[i] < population[j]:
                        self.topology[i, j] = 1
                    elif population[i] > population[j]:
                        self.topology[j, i] = 1
                    else:
                        self.topology[i, j] = 0.5
                        self.topology[j, i] = 0.5

            # Update adaptive neighborhood
            for i in range(self.dim):
                for j in range(self.dim):
                    if self.topology[i, j] == 1:
                        self.adaptive_neighborhood[i, j] = 1
                    elif self.topology[i, j] == 0.5:
                        self.adaptive_neighborhood[i, j] = 0.5
                    else:
                        self.adaptive_neighborhood[i, j] = 0

            population = population.flatten()

        return np.min(fitness)

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
opt = MultiDimensionalTopologyBasedEvolutionarySearch(budget, dim)
print(opt(func))