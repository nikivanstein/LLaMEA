import numpy as np

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, alpha=0.35):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))

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

# HybridEvolutionaryAlgorithm with Adaptive Sampling (HEAS)
# Description: This algorithm combines the adaptive sampling strategy of AEAS with the hybrid evolutionary algorithm, allowing for more efficient exploration in promising areas of the search space.

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, alpha=0.35):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = self.adaptive_sampling(func)(self.population)
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = self.adaptive_sampling(func)(self.population)
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = self.adaptive_sampling(func)(self.population)
            best_func = func(func_evals)
            if np.any(best_func!= func(func_evals)):
                func_evals = self.adaptive_sampling(func)(self.population)
            if np.all(best_func == func(func_evals)):
                break
            num_evaluations += 1
        return func_evals