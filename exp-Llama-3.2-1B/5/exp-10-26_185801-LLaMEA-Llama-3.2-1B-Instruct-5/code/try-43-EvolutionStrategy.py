import numpy as np
import random
import copy

class EvolutionStrategy:
    def __init__(self, budget, dim, mutation_rate, elite_size, mutation_probability, crossover_probability):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
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
            if func_evaluations >= self.budget:
                break
            new_individual = self.evaluate_fitness(new_individual)
            if np.isnan(new_individual) or np.isinf(new_individual):
                new_individual = copy.deepcopy(self.search_space)
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if random.random() < self.mutation_probability:
                new_individual = self.mutation(new_individual)
            if random.random() < self.mutation_probability:
                new_individual = self.mutation(new_individual)
            if random.random() < self.crossover_probability:
                new_individual = self.crossover(new_individual)
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return new_individual

    def mutation(self, individual):
        new_individual = copy.deepcopy(individual)
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                new_individual[i] += random.uniform(-1, 1)
        return new_individual

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_probability:
            crossover_point = random.randint(1, self.dim-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child
        return parent1

class HEBBO(EvolutionStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim, mutation_rate=0.1, elite_size=10, mutation_probability=0.1, crossover_probability=0.5)

    def evaluate_fitness(self, individual):
        func_value = self.func(individual)
        return func_value

class HBBP(EvolutionStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim, mutation_rate=0.1, elite_size=10, mutation_probability=0.1, crossover_probability=0.5)

    def func(self, individual):
        func_value = individual
        return func_value

class HBE(EvolutionStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim, mutation_rate=0.1, elite_size=10, mutation_probability=0.1, crossover_probability=0.5)

    def func(self, individual):
        func_value = individual
        return func_value

class HBEQ(EvolutionStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim, mutation_rate=0.1, elite_size=10, mutation_probability=0.1, crossover_probability=0.5)

    def func(self, individual):
        func_value = individual
        return func_value

# Description: Evolutionary Black Box Optimization using Evolution Strategies
# Code: 