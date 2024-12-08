import random
import numpy as np

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
        self.algorithm = Metaheuristic(budget, dim)

    def __call__(self, func):
        # Refine the strategy by changing the individual lines
        for _ in range(10):
            new_individual = self.algorithm.evaluate_fitness(lambda x: x + 0.1 * random.choice([1, -1]))(func)

        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(new_individual))

        # Select the best function value
        best_func = max(set([func(x) for x in new_individual]), key=func)

        # Update the search space
        self.algorithm.search_space = [x for x in self.algorithm.search_space if x not in best_func]

        return best_func

# Test the algorithm
def func(x):
    return np.sin(x)

algorithm = NovelMetaheuristicAlgorithm(budget=1000, dim=5)
best_func = algorithm(func)
print("Best function:", best_func)