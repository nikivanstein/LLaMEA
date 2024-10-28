import numpy as np
import random

class ABSEC:
    def __init__(self, budget, dim, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate
        self.population_size = 100

    def __call__(self, func):
        population = [func(np.random.uniform(self.search_space)) for _ in range(self.population_size)]
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(i+1, self.population_size):
                    if random.random() < self.mutation_rate:
                        population[i] += (population[j] - population[i]) * (random.random() - 0.5)
            population = sorted(population, key=func, reverse=True)
            func_value = population[0]
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def adapt_boundaries(self, func):
        boundaries = [np.linspace(-5.0, 5.0, self.dim), np.linspace(-5.0, 5.0, self.dim)]
        while True:
            new_boundaries = []
            for i in range(self.dim):
                new_boundaries.append(np.linspace(-5.0, 5.0, self.dim)[i] + random.uniform(-1, 1) * (self.search_space[i] + random.uniform(-1, 1)))
            new_func_value = func(np.array(new_boundaries))
            if np.abs(new_func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.search_space = new_boundaries
            func_value = new_func_value
        return func_value