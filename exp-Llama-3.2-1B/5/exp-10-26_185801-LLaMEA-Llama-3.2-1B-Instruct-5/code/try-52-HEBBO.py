import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.evolutionary_strategy = None

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
        if random.random() < 0.05:
            self.evolutionary_strategy = random.choice(['max','min', 'uniform'])
        if self.evolutionary_strategy =='max':
            return individual + random.uniform(0, 1)
        elif self.evolutionary_strategy =='min':
            return individual - random.uniform(0, 1)
        else:
            return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            return np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
        else:
            return np.concatenate((parent2[:self.dim//2], parent1[self.dim//2:]))

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

class GeneticAlgorithm(HEBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def fitness(self, individual):
        return self.evaluate_fitness(individual)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            individual = random.uniform(self.search_space)
            new_individual = self.mutate(individual)
            new_individual = self.crossover(new_individual, individual)
            self.func_evaluations += 1
            func_value = self.evaluate_fitness(new_individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

# Description: Black Box Optimization using Genetic Algorithm with Evolved Strategy
# Code: 