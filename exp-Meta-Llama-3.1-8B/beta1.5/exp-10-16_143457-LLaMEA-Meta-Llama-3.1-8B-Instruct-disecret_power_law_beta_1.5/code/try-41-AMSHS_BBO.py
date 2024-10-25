import numpy as np
import random

class AMSHS_BBO:
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
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.rand = random.Random()

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

    def _crossover(self, x1, x2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if self.rand.random() < self.crossover_rate:
                child[i] = (x1[i] + x2[i]) / 2
            else:
                child[i] = x1[i]
        return child

    def _mutation(self, x):
        for i in range(self.dim):
            if self.rand.random() < self.mutation_rate:
                x[i] += self.rand.uniform(-1, 1)
                x[i] = np.clip(x[i], self.lower_bound, self.upper_bound)
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
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                population[j] = x

            for k in range(self.swarm_size):
                x1 = self.rand.uniform(self.lower_bound, self.upper_bound, self.dim)
                x2 = self.rand.uniform(self.lower_bound, self.upper_bound, self.dim)
                x1 = self._harmony_search(x1, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                x2 = self._harmony_search(x2, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                child = self._crossover(x1, x2)
                child = self._mutation(child)
                fitness = self._evaluate_function(child)
                if fitness < best_fitness:
                    best_solution = child
                    best_fitness = fitness
                harmony_memory = self._update_harmony_memory(child, harmony_memory)

        return best_solution, best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 1000
dim = 10
amshs_bbo = AMSHS_BBO(budget, dim)
best_solution, best_fitness = amshs_bbo(func)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)