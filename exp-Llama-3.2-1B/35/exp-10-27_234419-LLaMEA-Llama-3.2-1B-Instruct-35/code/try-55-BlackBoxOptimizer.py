import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = np.inf
                for i in range(self.sample_size):
                    new_individual = self.evaluate_fitness(self.sample_indices + np.array([i]) / self.sample_size * self.search_space)
                    if new_individual < best_func:
                        best_func = new_individual
                        self.sample_indices = np.array([i]) / self.sample_size * self.search_space
                self.sample_indices = self.sample_indices[:self.sample_size]
            else:
                best_func = np.inf
                for i in range(self.sample_size):
                    new_individual = self.evaluate_fitness(self.sample_indices + np.array([i]) / self.sample_size * self.search_space)
                    if new_individual < best_func:
                        best_func = new_individual
                        self.sample_indices = np.array([i]) / self.sample_size * self.search_space
                self.sample_indices = self.sample_indices[:self.sample_size]

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)

def evaluateBBOB(func, individual, logger):
    updated_individual = func(individual)
    logger.update_fitness(updated_individual, 1)
    return updated_individual

def mutation_exp(individual, logger):
    new_individual = individual + np.random.normal(0, 0.1, individual.shape)
    logger.update_fitness(new_individual, 0.5)
    return new_individual

# Description: Adaptive Random Search with Adaptive Sampling and Local Search
# Code: 