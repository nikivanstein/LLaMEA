# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            for _ in range(self.budget):
                if np.linalg.norm(func(self.search_space[np.random.randint(0, self.search_space.shape[0])])) < self.budget / 2:
                    return self.search_space[np.random.randint(0, self.search_space.shape[0])]
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            self.search_space = np.vstack((self.search_space, self.search_space[np.random.randint(0, self.search_space.shape[0])]))
            self.search_space = np.delete(self.search_space, 0, axis=0)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
def __call__(self, func, budget=1000, iterations=1000, step_size=0.1, alpha=0.1):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization.

    Parameters:
    func (function): The black box function to optimize.
    budget (int, optional): The maximum number of function evaluations. Defaults to 1000.
    iterations (int, optional): The number of iterations for the hill climbing search. Defaults to 1000.
    step_size (float, optional): The step size for the hill climbing search. Defaults to 0.1.
    alpha (float, optional): The alpha parameter for the hill climbing search. Defaults to 0.1.

    Returns:
    individual (numpy array): The optimized individual.
    """
    while True:
        # Perform hill climbing search
        new_individual = self.evaluate_fitness(func, budget, iterations, step_size, alpha)
        # Refine the strategy based on the fitness
        if np.linalg.norm(func(new_individual)) < self.budget / 2:
            return new_individual
        # Update the search space
        self.search_space = np.vstack((self.search_space, new_individual))
        self.search_space = np.delete(self.search_space, 0, axis=0)