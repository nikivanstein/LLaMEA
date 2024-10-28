import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.best_individual = None
        self.best_fitness = None
        self.population = []

    def __call__(self, func, budget):
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

        if self.budget <= 1:
            self.population.append((self.sample_indices, func(self.sample_indices)))

        if self.budget <= 10:
            self.population = self.population[:self.budget]

        if np.abs(best_fitness - func(self.best_individual)) < 1e-6:
            self.best_individual = self.sample_indices
            self.best_fitness = func(self.best_individual)
        else:
            self.best_individual = None
            self.best_fitness = None

        return self.best_individual

    def mutate(self, individual):
        if self.best_individual is None:
            self.best_individual = individual
            self.best_fitness = func(individual)
        else:
            if np.random.rand() < 0.35:
                self.best_individual = individual
                self.best_fitness = func(individual)
            else:
                self.best_individual = self.best_individual[:self.sample_size]
                self.best_individual = self.best_individual[self.sample_size:]
        return individual

    def evaluate_fitness(self, individual):
        return func(individual)

# One-line description with the main idea
# Adaptive Black Box Optimization using Adaptive Random Search with Adaptive Sampling and Local Search