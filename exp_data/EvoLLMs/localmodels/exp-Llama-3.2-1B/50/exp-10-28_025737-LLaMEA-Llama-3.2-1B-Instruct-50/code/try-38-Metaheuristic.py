import random
import numpy as np
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

class BBOB:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, new_individual):
        # Evaluate the new individual
        func_values = [self.func(x) for x in new_individual]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class Mutation:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, new_individual):
        # Select a random individual from the search space
        individual = copy.deepcopy(self.search_space[0])

        # Randomly mutate the individual
        if random.random() < 0.45:
            # Randomly select a mutation point
            mutation_point = random.randint(0, self.dim - 1)

            # Randomly change the value at the mutation point
            individual[mutation_point] += random.uniform(-1.0, 1.0)

        # Evaluate the new individual
        func_values = [self.func(x) for x in new_individual]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class Selection:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim

    def __call__(self, new_individual):
        # Evaluate the new individual
        func_values = [self.func(x) for x in new_individual]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 