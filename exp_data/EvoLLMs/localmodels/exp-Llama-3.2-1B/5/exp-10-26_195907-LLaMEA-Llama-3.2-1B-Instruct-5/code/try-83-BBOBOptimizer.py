# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            self.population = self.population[:self.budget]
            self.population.append(x)

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def mutate(self, individual):
        while True:
            mutation = np.random.uniform(-1, 1, size=self.dim)
            mutated_individual = individual + mutation
            if np.linalg.norm(self.func(mutated_individual)) < self.budget / 2:
                return mutated_individual
            mutated_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            mutated_individual = np.vstack((mutated_individual, mutated_individual))
            mutated_individual = np.delete(mutated_individual, 0, axis=0)
            if np.linalg.norm(self.func(mutated_individual)) < self.budget / 2:
                return mutated_individual

# Novel Metaheuristic Algorithm for Black Box Optimization