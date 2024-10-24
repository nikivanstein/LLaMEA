import numpy as np
import random
from operator import itemgetter

class HybridMultiDimensionalTopologyBasedEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.topology = self.initialize_topology()

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

            # Hybrid approach: Use a subset of the best individuals to guide the evolution
            best_individuals = population[:int(self.dim/5)]
            best_individuals = np.array([itemgetter(0)(sorted(best_individuals, key=itemgetter(1)))[i] for i in range(int(self.dim/5))])
            new_population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, 1))
            for i in range(int(self.dim/5)):
                new_population[i] = best_individuals[i]
            new_population = np.c_[new_population, population[-int(self.dim/5):]]
            population = new_population

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

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
opt = HybridMultiDimensionalTopologyBasedEvolutionarySearch(budget, dim)
print(opt(func))