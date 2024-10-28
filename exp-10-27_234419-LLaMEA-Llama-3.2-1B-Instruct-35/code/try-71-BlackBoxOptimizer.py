import numpy as np
from scipy.optimize import minimize

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
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)

def bbob_optimize(func, budget, dim, initial_individual, mutation_rate, local_search_rate, num_iterations):
    optimizer = BlackBoxOptimizer(budget, dim)
    best_individual = initial_individual
    best_fitness = func(best_individual)
    best_individual = initial_individual
    best_fitness = func(best_individual)

    for _ in range(num_iterations):
        if local_search_rate > np.random.rand():
            optimizer(local_search_rate)

        # Evaluate the fitness of the current individual
        fitness = func(best_individual)

        # Refine the strategy based on the fitness
        if fitness < best_fitness:
            best_individual = best_individual
            best_fitness = fitness

        # Refine the strategy based on the probability of mutation
        if np.random.rand() < mutation_rate:
            # Perform mutation on the current individual
            mutated_individual = np.random.choice(self.search_space, size=self.dim, replace=False)
            mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
            mutated_individual = optimizer.evaluate_fitness(mutated_individual)

            # Update the best individual if the mutation leads to a better fitness
            if mutated_individual < best_individual:
                best_individual = mutated_individual
                best_fitness = mutated_individual

    return best_individual, best_fitness

# Description: Adaptive Random Search with Adaptive Sampling and Local Search
# Code: 