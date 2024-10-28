import numpy as np

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
            # Adaptive sampling strategy
            sampling_strategy = np.random.choice([0, 1], size=(self.population_size, self.dim), p=[self.sampling_rate, 1 - self.sampling_rate])
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim), args=sampling_strategy)
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            # Adaptive sampling strategy
            sampling_strategy = np.random.choice([0, 1], size=(self.population_size, self.dim), p=[self.sampling_rate, 1 - self.sampling_rate])
            func_evals = func(np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim), args=sampling_strategy))
            best_func_evals = np.any(func_evals!= func(np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))))
            if num_evaluations < self.budget // 2:
                num_evaluations += 1
                if np.any(best_func_evals):
                    # If best function is not found, try a different sampling strategy
                    sampling_strategy = np.random.choice([0, 1], size=(self.population_size, self.dim), p=[self.sampling_rate, 1 - self.sampling_rate])
                    func_evals = func(np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim), args=sampling_strategy))
            else:
                break
        return func_evals

# Description: Adaptive Hybrid Evolutionary Algorithm with Adaptive Sampling
# Code: 