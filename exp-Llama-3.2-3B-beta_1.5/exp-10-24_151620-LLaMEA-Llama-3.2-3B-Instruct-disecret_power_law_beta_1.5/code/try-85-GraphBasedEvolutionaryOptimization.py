import numpy as np
import random
import networkx as nx

class GraphBasedEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.G = self.initialize_graph()
        self.population = self.initialize_population()

    def initialize_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.dim))
        for i in range(1, self.dim):
            for j in range(i):
                if random.random() < 0.5:
                    G.add_edge(i, j)
        return G

    def initialize_population(self):
        population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, 1))
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = np.array([func(x) for x in self.population.flatten()])
            indices = np.argsort(fitness)
            population = self.population[indices]
            new_population = population[:self.dim//2]
            new_population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim//2, 1))
            new_population = np.c_[new_population, population[self.dim//2:]]
            population = np.concatenate((new_population, population[:self.dim//2]))
            self.G = self.update_graph(population)
            self.population = self.update_population(population)
        return np.min(fitness)

    def update_graph(self, population):
        G = self.G.copy()
        for i in range(self.dim):
            for j in range(self.dim):
                if population[i] < population[j]:
                    G.add_edge(i, j)
                elif population[i] > population[j]:
                    G.remove_edge(i, j)
        return G

    def update_population(self, population):
        new_population = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, 1))
        for i in range(self.dim):
            for j in range(i):
                if self.G.has_edge(i, j):
                    new_population[i] = population[j]
        return new_population

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
opt = GraphBasedEvolutionaryOptimization(budget, dim)
print(opt(func))