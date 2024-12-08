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
        def evaluate_fitness(individual):
            new_individual = individual.copy()
            for _ in range(self.budget):
                new_individual = self.f(new_individual, func)
            return new_individual

        def f(individual, func):
            return func(individual)

        while True:
            new_individual = evaluate_fitness(random.choice(self.search_space))
            if np.linalg.norm(f(new_individual, self.func)) < self.budget / 2:
                return new_individual
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

def evaluateBBOB(func, budget, dim, individual, logger):
    new_individual = individual.copy()
    for _ in range(budget):
        new_individual = func(new_individual, func)
    return new_individual

def novel_metaheuristic_optimizer(budget, dim):
    return BBOBOptimizer(budget, dim)

# Example usage:
optimizer = novel_metaheuristic_optimizer(100, 2)
individual = np.random.uniform(-5.0, 5.0, size=(dim, 2))
optimizer(individual)

# Log the fitness of the individual
logger = logging.getLogger(__name__)
logger.info(optimizer(individual))