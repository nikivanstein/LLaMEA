import numpy as np
import random

class AMSHS_BBO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.swarm_size = 10
        self.harmony_memory_size = 10
        self.paranoid = 0.1
        self.bw = 0.1
        self.hsr = 0.1
        self.rand = random.Random()
        self.F = 0.5
        self.CR = 0.9

    def _generate_initial_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _evaluate_function(self, x):
        if self.budget > 0:
            self.budget -= 1
            return func(x)
        else:
            raise Exception("Budget exceeded")

    def _harmony_search(self, x, bounds):
        for i in range(self.dim):
            if self.rand.random() < self.hsr:
                x[i] = self.rand.uniform(bounds[i, 0], bounds[i, 1])
        return x

    def _update_harmony_memory(self, x, memory):
        if self.rand.random() < self.paranoid:
            memory[self.rand.randint(0, self.harmony_memory_size - 1)] = x
        return memory

    def _differential_evolution(self, x, y, z):
        for i in range(self.dim):
            r1 = self.rand.randint(0, self.population_size - 1)
            r2 = self.rand.randint(0, self.population_size - 1)
            if self.rand.random() < self.CR or i == self.rand.randint(0, self.dim - 1):
                x[i] = y[i] + self.F * (z[i] - y[i])
                if x[i] < self.lower_bound:
                    x[i] = self.lower_bound
                elif x[i] > self.upper_bound:
                    x[i] = self.upper_bound
        return x

    def __call__(self, func):
        population = self._generate_initial_population()
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        best_solution = np.zeros(self.dim)
        best_fitness = float('inf')

        for i in range(self.budget):
            for j in range(self.population_size):
                x = population[j]
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                y = population[self.rand.randint(0, self.population_size - 1)]
                z = population[self.rand.randint(0, self.population_size - 1)]
                x = self._differential_evolution(x, y, z)
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                population[j] = x

            for k in range(self.swarm_size):
                x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                y = population[self.rand.randint(0, self.population_size - 1)]
                z = population[self.rand.randint(0, self.population_size - 1)]
                x = self._differential_evolution(x, y, z)
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                harmony_memory = self._update_harmony_memory(x, harmony_memory)

        return best_solution, best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 1000
dim = 10
amshs_bbo_de = AMSHS_BBO_DE(budget, dim)
best_solution, best_fitness = amshs_bbo_de(func)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)