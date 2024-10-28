import numpy as np

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, adaptive_sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.adaptive_sampling_rate = adaptive_sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.best_func = None
        self.best_score = 0.0
        self.score = 0.0

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= best_func):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == best_func):
                break
        self.best_func = best_func
        self.best_score = np.mean(best_func)
        self.score = self.best_score
        return best_func

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
        self.best_func = best_func
        self.best_score = np.mean(best_func)
        self.score = self.best_score
        return best_func_evals