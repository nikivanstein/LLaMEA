import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class MutationExp:
    def __init__(self, mutation_rate, dim):
        self.mutation_rate = mutation_rate
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, individual, logger):
        new_individual = individual.copy()
        for _ in range(self.mutation_rate * self.budget):
            if np.random.rand() < self.mutation_rate:
                idx = np.random.choice(self.search_space.shape[0])
                new_individual[idx] = func(new_individual[idx])
        return new_individual

class SelectionExp:
    def __init__(self, num_parents, dim):
        self.num_parents = num_parents
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, individuals, logger):
        sorted_indices = np.argsort(func(individuals))
        selected_indices = sorted_indices[:self.num_parents]
        selected_individuals = individuals[selected_indices]
        return selected_individuals

class GeneticAlgorithmExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.mutation_rate = 0.01
        self.selection_rate = 0.1

    def __call__(self, func):
        population_size = 100
        population = np.random.uniform(low=self.search_space, high=self.search_space, size=(population_size, self.dim))
        for _ in range(self.budget):
            parents = SelectionExp(self.selection_rate, self.dim)(func, population, self.logger)
            offspring = MutationExp(self.mutation_rate, self.dim)(func, parents, self.logger)
            population = np.concatenate((population, offspring))
        return np.mean(population, axis=0)

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = np.random.uniform(low=self.search_space, high=self.search_space, size=(100, self.dim))

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.population = np.linspace(self.search_space, func_value, self.dim)
        return func_value

# Description: Novel Black Box Optimization Algorithm using Evolutionary Strategies
# Code: 