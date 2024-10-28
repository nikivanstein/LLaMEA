import numpy as np
import random

class HybridEvolutionaryAlgorithm:
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
        self.sample_count = 0

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if self.num_evaluations < self.budget:
                best_func = func(self.population)
                if np.any(best_func!= func(self.population)):
                    self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
            self.num_evaluations += 1
            self.best_func = best_func
            self.best_score = np.max(np.array(best_func))
        return func(self.population)

    def adaptive_sampling(self, func):
        if random.random() < self.sampling_rate:
            num_evaluations = random.randint(1, self.budget)
            while num_evaluations < self.budget:
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
                best_func = func(func_evals)
                if np.any(best_func!= func(func_evals)):
                    func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
                if np.all(best_func == func(func_evals)):
                    break
                num_evaluations += 1
            self.sample_count += num_evaluations
            return func_evals
        else:
            return func(self.population)