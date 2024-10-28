import numpy as np

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, adaptive_rate=0.35):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.adaptive_rate = adaptive_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.best_func = None

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= best_func):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == best_func):
                break
        return func(self.population)

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            if np.any(best_func!= best_func):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == best_func):
                break
            num_evaluations += 1
        return func_evals

    def adapt(self):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = self.adaptive_sampling(func)
            if num_evaluations == 0:
                best_func = func
                break
            if np.mean(np.abs(func_evals - best_func)) < self.adaptive_rate * np.std(func_evals):
                best_func = func_evals
                break
            num_evaluations += 1
        return best_func

    def __str__(self):
        return f"AdaptiveHybridEvolutionaryAlgorithm: Adaptive Sampling with Adaptive Rate {self.adaptive_rate}"

# Description: Adaptive Hybrid Evolutionary Algorithm with Adaptive Sampling
# Code: 