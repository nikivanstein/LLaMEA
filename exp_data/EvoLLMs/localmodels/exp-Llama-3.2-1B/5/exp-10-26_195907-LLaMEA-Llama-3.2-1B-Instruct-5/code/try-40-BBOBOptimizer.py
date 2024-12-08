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
        
        while True:
            for _ in range(self.budget):
                individual = random.choice(self.search_space)
                if evaluate_fitness(individual) >= self.budget / 2:
                    return individual
            new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 