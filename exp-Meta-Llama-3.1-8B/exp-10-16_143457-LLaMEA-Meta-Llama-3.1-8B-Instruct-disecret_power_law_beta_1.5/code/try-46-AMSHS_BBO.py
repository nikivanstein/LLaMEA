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
        self.rand = random.Random()
        self.de_rand = np.random.rand(self.population_size, self.dim)
        self.de_f = 0.5
        self.cauchy_lambda = 1.0

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

    def _mutation_de(self, x):
        r1, r2, r3 = self.rand.randint(0, self.population_size - 1), self.rand.randint(0, self.population_size - 1), self.rand.randint(0, self.population_size - 1)
        v = x + self.de_f * (self.de_rand[r1] - self.de_rand[r2]) + self.de_f * (self.de_rand[r3] - self.de_rand[r2])
        v = v + self.cauchy_lambda * np.random.standard_cauchy(self.dim)
        return v

    def _mutation_cauchy(self, x):
        v = x + self.cauchy_lambda * np.random.standard_cauchy(self.dim)
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
                v1 = self._mutation_de(x)
                v2 = self._mutation_cauchy(x)
                x = (v1 + v2) / 2
                x = np.clip(x, self.lower_bound, self.upper_bound)
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                population[j] = x

            for k in range(self.swarm_size):
                x = self.rand.uniform(self.lower_bound, self.upper_bound, self.dim)
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                v1 = self._mutation_de(x)
                v2 = self._mutation_cauchy(x)
                x = (v1 + v2) / 2
                x = np.clip(x, self.lower_bound, self.upper_bound)
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
amshs_bbo = AMSHS_BBO(budget, dim)
best_solution, best_fitness = amshs_bbo(func)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)