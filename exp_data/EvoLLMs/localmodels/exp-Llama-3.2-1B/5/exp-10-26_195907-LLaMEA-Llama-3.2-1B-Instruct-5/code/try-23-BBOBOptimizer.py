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

    def __call__(self, func, budget):
        if budget <= 0:
            raise ValueError("Budget must be greater than zero")

        def evaluate_fitness(individual, budget):
            new_individual = individual
            while True:
                for _ in range(budget):
                    new_individual = self.evaluate_new_individual(new_individual)
                    if np.linalg.norm(func(new_individual)) < budget / 2:
                        return new_individual
                new_individual = self.evaluate_new_individual(new_individual)
            return new_individual

        def evaluate_new_individual(individual):
            return self.search_space[np.random.randint(0, self.search_space.shape[0]), :]

        self.search_space = np.vstack((self.search_space, evaluate_fitness(evaluate_fitness, self.budget, func)))

def evaluateBBOB(func, budget, func_evals, func_evals_limit):
    optimizer = BBOBOptimizer(budget, func_evals)
    return optimizer(func, func_evals)

# Example usage:
def func(x):
    return np.sum(x)

budget = 1000
func_evals = 1000
func_evals_limit = 500
result = evaluateBBOB(func, budget, func_evals, func_evals_limit)
print(result)