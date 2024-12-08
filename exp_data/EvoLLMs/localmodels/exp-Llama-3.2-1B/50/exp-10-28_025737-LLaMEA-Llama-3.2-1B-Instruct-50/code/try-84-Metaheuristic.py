# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from collections import deque

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

        # Refine the strategy using probability 0.45
        if random.random() < 0.45:
            # Randomly swap two elements in the search space
            i = random.randint(0, dim - 1)
            j = random.randint(0, dim - 1)
            self.search_space[i], self.search_space[j] = self.search_space[j], self.search_space[i]

        return best_func

class BBOB:
    def __init__(self, func, budget):
        self.func = func
        self.budget = budget
        self.population = deque([Metaheuristic(budget, 10) for _ in range(10)])

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.population[0].search_space = [x for x in self.population[0].search_space if x not in best_func]

        # Refine the strategy using probability 0.45
        if random.random() < 0.45:
            # Randomly swap two elements in the search space
            i = random.randint(0, self.population[0].dim - 1)
            j = random.randint(0, self.population[0].dim - 1)
            self.population[0].search_space[i], self.population[0].search_space[j] = self.population[0].search_space[j], self.population[0].search_space[i]

        return best_func

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
bboo = BBOB(lambda x: np.sin(x), 100)
print(bboo(func=np.sin))  # prints a random function from the BBOB test suite