import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        def evaluate_fitness(individual):
            return self.func(individual)

        def __call__(self, func, budget=100):
            while True:
                for _ in range(budget):
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                    if np.linalg.norm(func(x)) < budget / 2:
                        return x
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                self.search_space = np.vstack((self.search_space, x))
                self.search_space = np.delete(self.search_space, 0, axis=0)

        return __call__(func, budget)

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# The algorithm uses a novel metaheuristic approach that combines exploration and exploitation to optimize the black box function.
# The search space is varied between -5.0 and 5.0, and the budget is used to limit the number of function evaluations.
# The algorithm refines its strategy based on the fitness values of the individuals in the population.