import numpy as np
from collections import deque
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = deque(maxlen=self.budget)

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
            i, j = random.sample(range(self.dim), 2)
            self.search_space[i] = random.uniform(-5.0, 5.0)
            self.search_space[j] = random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        i, j = random.sample(range(self.dim), 2)
        child = parent1[:i] + parent2[i:j] + parent1[j:]
        return child

    def evaluate_fitness(self, individual):
        func_value = self.func_evaluations(individual, self.search_space)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def select_solution(self):
        if random.random() < 0.5:
            return random.choice(self.population)
        else:
            return self.population.popleft()

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using population-based search and mutation with crossover and selection strategies.