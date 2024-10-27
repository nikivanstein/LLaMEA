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
        def evaluate_fitness(individual, budget):
            for _ in range(budget):
                x = individual
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                x = np.vstack((x, individual))
                x = np.delete(x, 0, axis=0)
            return individual

        population = [evaluate_fitness(individual, self.budget) for individual in random.sample(self.search_space, self.budget)]
        return BBOBOptimizer(self.budget, self.dim).evaluate_fitness(np.mean(population), self.budget)

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BBOBOptimizer(budget, dim)

    def __call__(self, func):
        while True:
            individual = self.optimizer.func(np.random.uniform(-5.0, 5.0, size=(self.dim, 2)))
            updated_individual = self.optimizer.evaluate_fitness(individual, self.optimizer.budget)
            if np.linalg.norm(func(updated_individual)) < self.optimizer.budget / 2:
                return updated_individual
            self.optimizer.search_space = np.vstack((self.optimizer.search_space, updated_individual))
            self.optimizer.search_space = np.delete(self.optimizer.search_space, 0, axis=0)

# Test the algorithm
optimizer = NovelMetaheuristicOptimizer(100, 5)
func = lambda x: np.sum(x)
print(optimizer(__call__(func)))