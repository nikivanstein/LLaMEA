# HybridEvolutionaryAlgorithm: "Hybrid Evolutionary Algorithm with Adaptive Sampling and Adaptive Mutation" (HESA)
# Code: 
# ```python
import numpy as np

class HESA:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, adaptive_sampling_rate=0.1, adaptive Mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.adaptive_sampling_rate = adaptive_sampling_rate
        self.adaptive_Mutation_rate = adaptive_Mutation_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.best_func = None
        self.best_score = np.inf
        self.best_individual = None

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        self.best_func = best_func
        self.best_score = np.mean(np.abs(best_func - func(self.population)))
        if self.best_score < self.best_score:
            self.best_score = self.best_score
            self.best_individual = self.population

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func_evals = func(func_evals)
            if np.any(best_func_evals!= best_func):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func_evals == best_func):
                break
            num_evaluations += 1
        self.best_func_evals = best_func_evals
        self.best_individual = best_func_evals
        self.best_score = np.mean(np.abs(best_func_evals - func(best_func_evals)))
        if self.best_score < self.best_score:
            self.best_score = self.best_score
        return best_func_evals

    def adaptive_mutation(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func_evals = func(func_evals)
            if np.any(best_func_evals!= best_func):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func_evals == best_func):
                break
            num_evaluations += 1
        self.best_func_evals = best_func_evals
        self.best_individual = best_func_evals
        self.best_score = np.mean(np.abs(best_func_evals - func(best_func_evals)))
        if self.best_score < self.best_score:
            self.best_score = self.best_score
        return best_func_evals

# One-line description with the main idea
# HybridEvolutionaryAlgorithm with Adaptive Sampling and Adaptive Mutation
# Code: 
# ```python
# HESA: Hybrid Evolutionary Algorithm with Adaptive Sampling and Adaptive Mutation
# ```
# ```python
# import numpy as np

class HESA:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, adaptive_sampling_rate=0.1, adaptive_Mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.adaptive_sampling_rate = adaptive_sampling_rate
        self.adaptive_Mutation_rate = adaptive_Mutation_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.best_func = None
        self.best_score = np.inf
        self.best_individual = None

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        self.best_func = best_func
        self.best_score = np.mean(np.abs(best_func - func(self.population)))
        if self.best_score < self.best_score:
            self.best_score = self.best_score
            self.best_individual = self.population

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func_evals = func(func_evals)
            if np.any(best_func_evals!= best_func):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func_evals == best_func):
                break
            num_evaluations += 1
        self.best_func_evals = best_func_evals
        self.best_individual = best_func_evals
        self.best_score = np.mean(np.abs(best_func_evals - func(best_func_evals)))
        if self.best_score < self.best_score:
            self.best_score = self.best_score
        return best_func_evals

    def adaptive_mutation(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func_evals = func(func_evals)
            if np.any(best_func_evals!= best_func):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func_evals == best_func):
                break
            num_evaluations += 1
        self.best_func_evals = best_func_evals
        self.best_individual = best_func_evals
        self.best_score = np.mean(np.abs(best_func_evals - func(best_func_evals)))
        if self.best_score < self.best_score:
            self.best_score = self.best_score
        return best_func_evals

# One-line description with the main idea
# HybridEvolutionaryAlgorithm with Adaptive Sampling and Adaptive Mutation
# Code: 
# ```python
# HESA: Hybrid Evolutionary Algorithm with Adaptive Sampling and Adaptive Mutation
# ```