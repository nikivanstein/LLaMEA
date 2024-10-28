import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            if np.any(best_func!= func(func_evals)):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(func_evals)):
                break
            num_evaluations += 1
        return func_evals

    def adaptive_hybrid(self, func, mutation_rate, sampling_rate):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = self.adaptive_sampling(func)
            if np.any(func_evals!= func_evals):
                func_evals = self.adaptive_sampling(func)
            if np.all(func_evals == func_evals):
                break
            num_evaluations += 1
        return func_evals

    def __str__(self):
        return "Adaptive Hybrid Evolutionary Algorithm"

# Description: Adaptive Hybrid Evolutionary Algorithm with Adaptive Sampling
# Code: 