import numpy as np

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.best_func = None
        self.best_score = float('-inf')
        self.num_evaluations = 0

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= best_func):  
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == best_func):
                break
        self.best_func = best_func
        self.best_score = np.max(np.all(best_func == func(self.population), axis=1))
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

    def adaptive_hypermutation(self, func, mutation_rate=0.01):
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

    def adaptive_crossover(self, func, crossover_rate=0.5):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals1 = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            func_evals2 = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = np.max(np.all(func_evals1==func(np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))), axis=1) & np.all(func_evals2 == func(np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))), axis=1))
            if np.any(best_func!= best_func):  
                func_evals1 = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
                func_evals2 = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == best_func):
                break
            num_evaluations += 1
        return func_evals1, func_evals2

# Description: Adaptive Hybrid Evolutionary Algorithm with Adaptive Sampling and Adaptive Mutation
# Code: 