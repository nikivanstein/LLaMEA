import numpy as np
import random

class MultiDimensionalTopologyBasedEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.topology = self.initialize_topology()
        self.population = self.initialize_population()

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

    def initialize_population(self):
        population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, 1))
        return population

    def genetic_operator(self, population, topology):
        new_population = population[:self.dim//2]
        new_population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim//2, 1))
        new_population = np.c_[new_population, population[self.dim//2:]]
        population = np.concatenate((new_population, population[:self.dim//2]))
        return population

    def topology_based_selection(self, population):
        fitness = np.array([func(x) for x in population.flatten()])
        indices = np.argsort(fitness)
        return population[indices]

    def hybrid_selection(self, population, topology):
        new_population = self.topology_based_selection(population)
        new_topology = self.update_topology(new_population)
        for i in range(self.dim):
            for j in range(self.dim):
                if new_population[i] < new_population[j]:
                    new_topology[i, j] = 1
                elif new_population[i] > new_population[j]:
                    new_topology[j, i] = 1
        return new_population, new_topology

    def update_topology(self, population):
        new_topology = self.initialize_topology()
        for i in range(self.dim):
            for j in range(self.dim):
                if population[i] < population[j]:
                    new_topology[i, j] = 1
                elif population[i] > population[j]:
                    new_topology[j, i] = 1
        return new_topology

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = self.genetic_operator(self.population, self.topology)
            self.population = self.topology_based_selection(self.population)
            self.population, self.topology = self.hybrid_selection(self.population, self.topology)
            fitness = np.array([func(x) for x in self.population.flatten()])
            self.topology = self.update_topology(self.population)
            self.population = self.population.flatten()
        return np.min(fitness)

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
opt = MultiDimensionalTopologyBasedEvolutionarySearch(budget, dim)
print(opt(func))