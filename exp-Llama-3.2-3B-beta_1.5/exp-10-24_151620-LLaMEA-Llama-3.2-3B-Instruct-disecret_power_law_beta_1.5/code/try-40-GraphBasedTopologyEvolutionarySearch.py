import numpy as np
import random
import networkx as nx

class GraphBasedTopologyEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.dim))
        self.topology = self.initialize_topology()

    def initialize_topology(self):
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                if random.random() < 0.5:
                    self.G.add_edge(i, j)
                    self.G.add_edge(j, i)
        return self.G

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
        return np.min(fitness)

    def update_topology(self, population):
        new_topology = self.initialize_topology()
        for i in range(self.dim):
            for j in range(self.dim):
                if population[i] < population[j]:
                    new_topology.add_edge(i, j)
                elif population[i] > population[j]:
                    new_topology.add_edge(j, i)
        return new_topology

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
opt = GraphBasedTopologyEvolutionarySearch(budget, dim)
print(opt(func))