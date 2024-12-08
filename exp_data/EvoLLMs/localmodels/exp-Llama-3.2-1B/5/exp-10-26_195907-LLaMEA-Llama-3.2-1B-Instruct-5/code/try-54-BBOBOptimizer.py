import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)
        self.population = []

    def __call__(self, func):
        while True:
            for _ in range(self.budget):
                new_individual = self.evaluate_fitness(self.population[-1])
                updated_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])][0]
                self.population.append(updated_individual)
                self.search_space = np.vstack((self.search_space, updated_individual))
                self.search_space = np.delete(self.search_space, 0, axis=0)
                if np.linalg.norm(func(updated_individual)) < self.budget / 2:
                    return updated_individual
            self.population = np.array(self.population)
            self.population = self.population[:self.budget]
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def mutate(self, individual):
        return individual + np.random.uniform(-1, 1, size=self.dim)

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization\n"