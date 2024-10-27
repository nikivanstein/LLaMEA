import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            for _ in range(self.budget):
                if np.random.rand() < 0.05:
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                else:
                    x = self.search_space[np.random.choice(self.search_space.shape[0], dim, replace=False)]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)
        self.population = [BBOBOptimizer(self.budget, dim) for _ in range(100)]

    def __call__(self, func):
        while True:
            fitness_values = [individual.func(x) for individual in self.population for x in individual.search_space]
            selected_individuals = self.population[np.argsort(fitness_values)]
            selected_individuals = selected_individuals[:self.budget]
            new_individuals = []
            for _ in range(self.budget):
                if np.random.rand() < 0.05:
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                else:
                    x = self.search_space[np.random.choice(self.search_space.shape[0], dim, replace=False)]
                new_individual = selected_individuals[np.random.randint(0, len(selected_individuals))]
                new_individual.search_space = np.vstack((new_individual.search_space, x))
                new_individual.search_space = np.delete(new_individual.search_space, 0, axis=0)
                new_individuals.append(new_individual)
            self.population = new_individuals
            if np.linalg.norm(func(self.population[0].search_space[np.random.randint(0, self.population[0].search_space.shape[0])]) - func(self.population[0].func())) < self.budget / 2:
                return self.population[0].func()

# Code: 
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 