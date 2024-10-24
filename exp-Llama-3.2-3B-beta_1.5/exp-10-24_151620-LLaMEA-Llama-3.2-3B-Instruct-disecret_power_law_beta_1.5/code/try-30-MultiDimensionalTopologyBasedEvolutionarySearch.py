import numpy as np
import random

class MultiDimensionalTopologyBasedEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0]) * np.ones((dim,))
        self.topology = self.initialize_topology()
        self.probability = 0.03773584905660377

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

            # Probabilistic crossover
            for i in range(self.dim):
                if random.random() < self.probability:
                    crossover_point = random.randint(0, self.dim - 1)
                    new_individual = np.concatenate((new_population[i, :crossover_point], population[i, crossover_point:], new_population[i, :crossover_point]))
                    population[i] = new_individual

            self.topology = self.update_topology(population)
            population = population.flatten()
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