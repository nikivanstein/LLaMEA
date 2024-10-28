import random
import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        for _ in range(self.budget):
            func = self.evaluate_func(func)
            if random.random() < self.mutation_rate:
                func = self.mutate_func(func)
        return func

    def evaluate_func(self, func):
        return np.linalg.norm(func - np.array([0.0] * self.dim))

    def mutate_func(self, func):
        idx = random.randint(0, self.dim - 1)
        func[idx] += random.uniform(-1.0, 1.0)
        return func

# Description: Bayesian Optimization with Adaptive Sampling
# Code: 