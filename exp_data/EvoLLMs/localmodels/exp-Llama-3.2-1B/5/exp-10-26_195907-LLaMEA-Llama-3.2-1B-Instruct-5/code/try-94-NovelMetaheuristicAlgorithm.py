import random
import numpy as np
from scipy.optimize import minimize

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, initial_individual, logger):
        while True:
            for _ in range(self.budget):
                new_individual = self.evaluate_fitness(initial_individual, func, logger)
                if np.linalg.norm(new_individual - initial_individual) < self.budget / 2:
                    return new_individual
                initial_individual = new_individual
            new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            logger.log(f"Individual: {new_individual}, Fitness: {np.linalg.norm(new_individual - initial_individual)}")

    def evaluate_fitness(self, individual, func, logger):
        return func(individual)

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 