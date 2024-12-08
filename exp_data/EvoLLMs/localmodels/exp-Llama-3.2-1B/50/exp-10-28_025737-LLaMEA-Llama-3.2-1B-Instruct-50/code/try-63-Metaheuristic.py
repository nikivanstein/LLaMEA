# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from collections import deque
import copy

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)
        self.population_size = 50
        self.population_history = deque(maxlen=100)

    def __call__(self, func):
        # Initialize the population
        population = [copy.deepcopy(self.metaheuristic) for _ in range(self.population_size)]

        # Evaluate the function a limited number of times
        num_evals = min(self.metaheuristic.budget, len(func(self.metaheuristic.search_space)))
        func_values = [func(x) for x in random.sample(self.metaheuristic.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.metaheuristic.search_space = [x for x in self.metaheuristic.search_space if x not in best_func]

        return best_func

# Test the algorithm
def test_func(x):
    return x**2 + 2*x + 1

algorithm = NovelMetaheuristicAlgorithm(budget=100, dim=10)
best_func = None
best_fitness = float('-inf')
for _ in range(100):
    best_func, best_fitness = algorithm(__call__(test_func))
    print(f"Best function: {best_func}, Best fitness: {best_fitness}")