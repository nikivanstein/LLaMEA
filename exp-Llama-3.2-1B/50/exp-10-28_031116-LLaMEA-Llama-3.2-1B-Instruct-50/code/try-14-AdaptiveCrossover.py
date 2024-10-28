import numpy as np
import random

class AdaptiveCrossover:
    def __init__(self, budget, dim, mutation_rate, crossover_probability):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_probability = crossover_probability
        self.population = []

    def __call__(self, func, population_size):
        # Initialize population with random functions
        for _ in range(population_size):
            func = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(func)

        # Evaluate functions until budget is exhausted
        for _ in range(self.budget):
            # Select two parents using roulette wheel selection
            parents = random.choices(self.population, weights=[1 / len(self.population)] * len(self.population), k=2)

            # Perform crossover
            child = parents[0][:self.dim // 2] + parents[1][self.dim // 2:]
            child = np.clip(child, -5.0, 5.0)

            # Evaluate child function
            child_score = np.linalg.norm(func - child)

            # Update population with best child
            if child_score > np.linalg.norm(func - self.population[0]):
                self.population[0] = child
                self.population.sort(key=lambda x: np.linalg.norm(x - func), reverse=True)

            # Apply mutation
            if random.random() < self.mutation_rate:
                mutation_index = random.randint(0, self.dim)
                self.population[mutation_index] += random.uniform(-1, 1)

    def fitness(self, func):
        # Evaluate function using the original black box function
        return np.linalg.norm(func - self.population[0])

# Description: Evolutionary Algorithm with Adaptive Step Size and Non-Adaptive Crossover
# Code: 