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

        # Update the search space using Iterated Permutation and Crossover
        self.search_space = [x for x in self.search_space if x not in best_func]
        self.search_space = self.iterated_permutation(self.search_space, 0.45)
        return best_func

    def iterated_permutation(self, search_space, p):
        """Iterated Permutation algorithm"""
        while len(search_space) > 0:
            # Select the next individual using Crossover
            parent1, parent2 = random.sample(search_space, 2)
            child = (parent1 + parent2) / 2
            # Select the next parent using Iterated Permutation
            parent = random.choices(search_space, p=p)[0]
            # Replace the parent with the child
            search_space.remove(parent)
            search_space.append(child)
        return search_space

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization using Iterated Permutation and Crossover
# Code: 