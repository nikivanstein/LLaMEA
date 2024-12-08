# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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
        self.search_space = self.search_space[:self.budget]

        return best_func

class NovelMetaheuristicAlgorithm:
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

        # Refine the strategy using probability 0.45
        new_individual = self.evaluate_fitness(best_func, func_values, 0.45)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in new_individual]
        self.search_space = self.search_space[:self.budget]

        return new_individual

    def evaluate_fitness(self, best_func, func_values, probability):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func_values))
        func_values = [func(x) for x in random.sample(func_values, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Refine the strategy using probability
        new_individual = [x for x in self.search_space if x not in best_func]
        new_individual = np.array(new_individual)
        new_individual = new_individual[np.argsort(func_values)]
        new_individual = new_individual[:self.budget]
        new_individual = new_individual[np.argsort(func_values)]

        return new_individual

# Test the algorithm
algorithm = NovelMetaheuristicAlgorithm(10, 5)
func = lambda x: np.sin(x)
best_func = algorithm(algorithm, func)
print(best_func)