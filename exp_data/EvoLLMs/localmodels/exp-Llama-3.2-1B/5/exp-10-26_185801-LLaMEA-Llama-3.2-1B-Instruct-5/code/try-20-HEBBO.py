import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('inf')

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

    def mutate(self, individual):
        if self.best_individual is None:
            self.best_individual = individual
            self.best_fitness = self.func(individual)
        else:
            new_individual = individual
            for i in range(self.dim):
                if np.random.rand() < 0.05:
                    new_individual[i] = np.random.uniform(self.search_space[i] - 1, self.search_space[i] + 1)
            if self.func_evaluations < self.budget:
                self.func_evaluations += 1
                new_fitness = self.func(new_individual)
                if new_fitness < self.best_fitness:
                    self.best_individual = new_individual
                    self.best_fitness = new_fitness
            else:
                new_individual = self.evaluate_fitness(new_individual)
                if new_individual > self.best_fitness:
                    self.best_individual = new_individual
                    self.best_fitness = new_individual
        return new_individual

    def evaluate_fitness(self, individual):
        func_value = self.func(individual)
        return func_value

    def select_next_individual(self):
        if self.best_individual is None:
            return None
        else:
            return np.random.choice(self.search_space, p=self.evaluate_fitness)

# Description: Black Box Optimization using Evolutionary Algorithm
# Code: 