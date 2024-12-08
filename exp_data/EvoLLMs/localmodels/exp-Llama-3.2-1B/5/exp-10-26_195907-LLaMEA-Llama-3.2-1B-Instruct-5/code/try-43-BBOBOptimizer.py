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
            for _ in range(min(self.budget, len(self.search_space))):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)
        self.population = [self.initialize_single() for _ in range(100)]

    def initialize_single(self):
        return np.random.uniform(-5.0, 5.0, size=(self.dim, 2))

    def __call__(self, func):
        while True:
            new_population = []
            for _ in range(min(self.budget, len(self.population))):
                new_individual = self.population[np.random.randint(0, len(self.population))]
                new_individual = self.optimizeBBOB(new_individual, func)
                new_population.append(new_individual)
            self.population = new_population

    def optimizeBBOB(self, individual, func):
        # Novel Metaheuristic Algorithm
        # Refine the individual lines of the selected solution to refine its strategy
        # The algorithm uses a probability of 0.05 to change the individual lines
        # to refine its strategy
        if np.random.rand() < 0.05:
            # Randomly change a line of the individual
            line_index = np.random.randint(0, self.dim)
            line = individual[line_index]
            new_line = line + random.uniform(-0.1, 0.1)
            individual[line_index] = new_line
        return individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# ```python
# BBOBOptimizer(1000, 10).__call__(BBOBOptimizer(1000, 10).func)
# ```python