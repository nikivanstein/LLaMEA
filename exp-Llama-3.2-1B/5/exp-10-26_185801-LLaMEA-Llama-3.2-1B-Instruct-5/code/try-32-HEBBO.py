import numpy as np
import random
from collections import deque

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

    def select_solution(self, population):
        if len(population) == 0:
            return None
        # Select a random parent from the population
        parent = random.choice(population)
        # Select a random child by mutation
        child = parent.copy()
        for _ in range(self.dim):
            if random.random() < 0.5:
                # Mutation: swap two random genes
                idx1, idx2 = random.sample(range(self.dim), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
        return child

    def mutate(self, individual):
        if random.random() < 0.05:
            # Mutation: swap two random genes
            idx1, idx2 = random.sample(range(self.dim), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        return func_value

class GeneticAlgorithm(HEBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        population = [self.select_solution([random.uniform(-5.0, 5.0) for _ in range(self.dim)]) for _ in range(100)]
        for _ in range(self.budget):
            new_population = []
            for parent in population:
                child = parent.copy()
                for _ in range(self.dim):
                    if random.random() < 0.05:
                        # Mutation: swap two random genes
                        idx1, idx2 = random.sample(range(self.dim), 2)
                        child[idx1], child[idx2] = child[idx2], child[idx1]
                new_population.append(child)
            population = new_population
        return max(population, key=self.evaluate_fitness)

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 