import numpy as np
import random

class AMSHS_DE:
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
        self.CR = 0.5

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

    def _differential_evolution(self, x, bounds):
        x1 = np.random.choice(self.population_size)
        x2 = np.random.choice(self.population_size)
        x3 = np.random.choice(self.population_size)

        v = x + self.F * (self.population[x1] - self.population[x2]) + self.CR * (self.population[x3] - x)
        v = np.clip(v, bounds[:, 0], bounds[:, 1])

        return v

    def __call__(self, func):
        population = self._generate_initial_population()
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        best_solution = np.zeros(self.dim)
        best_fitness = float('inf')

        for i in range(self.budget):
            for j in range(self.population_size):
                x = population[j]
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                x = self._differential_evolution(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                population[j] = x

            for k in range(self.swarm_size):
                x = self.rand.uniform(self.lower_bound, self.upper_bound, self.dim)
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                x = self._differential_evolution(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
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
amshs_de = AMSHS_DE(budget, dim)
best_solution, best_fitness = amshs_de(func)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)