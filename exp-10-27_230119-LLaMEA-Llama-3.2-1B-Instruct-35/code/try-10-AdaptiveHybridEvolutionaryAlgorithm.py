# Description: HybridEvolutionaryAlgorithm with Adaptive Sampling
# Code: 
# ```python
import numpy as np

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5, adaptive_sampling_rate=0.8):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.adaptive_sampling_rate = adaptive_sampling_rate

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

    def __str__(self):
        return f"HybridEvolutionaryAlgorithm with Adaptive Sampling (AOCC: {self.budget:.2f}, std: {self.budget**0.5:.2f})"

# Test the algorithm
hybrid_algorithm = AdaptiveHybridEvolutionaryAlgorithm(budget=100, dim=10)
func = np.random.uniform(-5.0, 5.0, size=(100, 10))
print(hybrid_algorithm(func))