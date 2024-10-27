import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def evaluate_fitness(individual):
            return self.func(individual)

        def mutate(individual):
            bounds = self.search_space[individual[:, 0]]
            new_individual = individual.copy()
            new_individual[:, 0] += random.uniform(-bounds[1], bounds[1])
            if np.random.rand() < 0.05:  # Refine strategy with probability 0.05
                new_individual[:, 1] = random.choice([-1, 1]) * (bounds[0] + random.uniform(-bounds[1], bounds[1]))
            return new_individual

        while True:
            fitness = evaluate_fitness(self.search_space[np.random.randint(0, self.search_space.shape[0])])
            if np.linalg.norm(func(self.search_space[np.random.randint(0, self.search_space.shape[0])])) < fitness:
                return mutate(self.search_space[np.random.randint(0, self.search_space.shape[0])])

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 