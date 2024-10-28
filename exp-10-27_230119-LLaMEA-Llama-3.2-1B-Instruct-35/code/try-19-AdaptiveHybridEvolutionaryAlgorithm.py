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
        self.best_func = None
        self.best_func_score = None

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        self.best_func = best_func
        self.best_func_score = func(self.best_func)

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = self.adaptive_sampling_helper(func, self.population, self.budget, self.sampling_rate)
            if np.any(func_evals!= func_evals):
                func_evals = self.adaptive_sampling_helper(func, self.population, self.budget, self.sampling_rate)
            if np.all(func_evals == func):
                break
            num_evaluations += 1
        return func_evals

    def adaptive_sampling_helper(self, func, population, budget, sampling_rate):
        results = []
        for _ in range(int(budget * sampling_rate)):
            func_evals = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
            func_results = func(func_evals)
            results.append((func_results, func_evals))
        return np.array(results)

    def run(self):
        if self.best_func_score is not None:
            return self.best_func_score
        else:
            return self.__call__(self.best_func)

# Description: Hybrid Evolutionary Algorithm with Adaptive Sampling
# Code: 