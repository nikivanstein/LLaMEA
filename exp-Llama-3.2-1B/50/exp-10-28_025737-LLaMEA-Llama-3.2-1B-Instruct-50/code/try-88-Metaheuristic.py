import numpy as np
import random

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

class MutationMetaheuristic(Metaheuristic):
    def __call__(self, func):
        # Select a random individual from the search space
        individual = random.choice(self.search_space)

        # Apply mutation to the individual
        mutated_individual = individual + random.uniform(-1, 1)

        # Evaluate the mutated individual
        mutated_func_values = [func(x) for x in random.sample(self.search_space, len(func(mutated_individual))))

        # Select the best mutated function value
        best_mutated_func = max(set(mutated_func_values), key=mutated_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_mutated_func]

        return best_mutated_func

class SelectionMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))

        # Select the best function value
        best_func = max(set(func(self.search_space)), key=func(self.search_space).count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class SelectionMutationMetaheuristic(SelectionMetaheuristic):
    def __call__(self, func):
        # Select a random individual from the search space
        individual = random.choice(self.search_space)

        # Apply mutation to the individual
        mutated_individual = individual + random.uniform(-1, 1)

        # Evaluate the mutated individual
        mutated_func_values = [func(x) for x in random.sample(self.search_space, len(func(mutated_individual))))

        # Select the best mutated function value
        best_mutated_func = max(set(mutated_func_values), key=mutated_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_mutated_func]

        return best_mutated_func

class ReplacementMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))

        # Select the best function value
        best_func = max(set(func(self.search_space)), key=func(self.search_space).count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class ReplacementMutationMetaheuristic(ReplacementMetaheuristic):
    def __call__(self, func):
        # Select a random individual from the search space
        individual = random.choice(self.search_space)

        # Apply mutation to the individual
        mutated_individual = individual + random.uniform(-1, 1)

        # Evaluate the mutated individual
        mutated_func_values = [func(x) for x in random.sample(self.search_space, len(func(mutated_individual))))

        # Select the best mutated function value
        best_mutated_func = max(set(mutated_func_values), key=mutated_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_mutated_func]

        return best_mutated_func

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 