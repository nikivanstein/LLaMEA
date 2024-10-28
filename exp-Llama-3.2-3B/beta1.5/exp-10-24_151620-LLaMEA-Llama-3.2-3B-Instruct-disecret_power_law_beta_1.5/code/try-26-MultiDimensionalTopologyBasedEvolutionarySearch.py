import numpy as np
import random

class MultiDimensionalTopologyBasedEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.topology = self.initialize_topology()
        self.mutation_prob = 0.018867924528301886

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

            # Adaptive mutation probability
            mutation_prob = self.mutation_prob * (1 - (fitness[0] / np.min(fitness)))
            for i in range(self.dim):
                if random.random() < mutation_prob:
                    new_x = population[i] + np.random.uniform(-1, 1)
                    if new_x < self.search_space[0]:
                        new_x = self.search_space[0]
                    elif new_x > self.search_space[1]:
                        new_x = self.search_space[1]
                    population[i] = new_x

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
opt = MultiDimensionalTopologyBasedEvolutionarySearch(budget, dim)
print(opt(func))