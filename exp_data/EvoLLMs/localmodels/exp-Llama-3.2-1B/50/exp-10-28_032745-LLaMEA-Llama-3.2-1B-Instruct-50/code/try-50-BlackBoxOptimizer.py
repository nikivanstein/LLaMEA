import numpy as np
from scipy.optimize import minimize
import random
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population = []
        self.logger = None

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            self.population.append([x, y, z] for x, y, z in zip(best_x, best_value, [1]*self.dim))
            self.logger = deque(maxlen=self.budget)
            for x, y, z in self.population:
                self.logger.append((x, y, z))
            if len(self.logger) == self.budget:
                self.logger.popleft()
        return self.population

    def select(self, population):
        return random.choices(population, weights=[x[2] for x in population], k=self.budget)

    def mutate(self, population):
        mutated_population = []
        for individual in population:
            new_x = [x + random.uniform(-0.01, 0.01) for x in individual]
            if random.random() < 0.5:
                new_x = [x - random.uniform(-0.01, 0.01) for x in new_x]
            mutated_population.append(new_x)
        return mutated_population

    def __repr__(self):
        return "BlackBoxOptimizer(budget={}, dim={})".format(self.budget, self.dim)

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 