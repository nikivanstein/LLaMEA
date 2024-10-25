import numpy as np
import random

class Adaptive_AMSHS_BBO:
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
        self.swarm_size_factor = 1.0
        self.harmony_memory_size_factor = 1.0

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

    def _update_swarm_size(self):
        self.swarm_size = int(self.population_size * self.swarm_size_factor)
        self.swarm_size_factor = min(2.0, max(0.1, self.swarm_size_factor + self.rand.random() * 0.1 - 0.05))

    def _update_harmony_memory_size(self):
        self.harmony_memory_size = int(self.population_size * self.harmony_memory_size_factor)
        self.harmony_memory_size_factor = min(2.0, max(0.1, self.harmony_memory_size_factor + self.rand.random() * 0.1 - 0.05))

    def __call__(self, func):
        population = self._generate_initial_population()
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        best_solution = np.zeros(self.dim)
        best_fitness = float('inf')
        iteration = 0

        while self.budget > 0:
            self._update_swarm_size()
            self._update_harmony_memory_size()
            for i in range(self.swarm_size):
                x = self.rand.uniform(self.lower_bound, self.upper_bound, self.dim)
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                harmony_memory = self._update_harmony_memory(x, harmony_memory)

            for j in range(self.population_size):
                x = population[j]
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                population[j] = x

            iteration += 1

        return best_solution, best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 1000
dim = 10
adaptive_amshs_bbo = Adaptive_AMSHS_BBO(budget, dim)
best_solution, best_fitness = adaptive_amshs_bbo(func)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)