import numpy as np

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.fitness_history = np.zeros((population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func, num_evaluations):
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            if np.any(best_func!= func(func_evals)):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(func_evals)):
                break
            num_evaluations += 1
        return func_evals

    def adaptive_hybrid(self, func, num_evaluations, adaptive_rate=0.5):
        if np.random.rand() < adaptive_rate:
            func_evals = self.adaptive_sampling(func, num_evaluations)
        else:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        return func(func_evals)

    def run(self, func, num_evaluations):
        best_func = func(self.population)
        best_fitness = np.mean(best_func)
        fitness_history = np.zeros((num_evaluations, self.dim))
        for i in range(num_evaluations):
            fitness_history[i] = np.mean(best_func)
            if np.any(best_func!= best_func):
                best_func = func(self.population)
                fitness_history[i] = np.mean(best_func)
            if np.all(best_func == best_func):
                break
        return best_func, best_fitness, fitness_history

# Description: Adaptive Hybrid Evolutionary Algorithm
# Code: 