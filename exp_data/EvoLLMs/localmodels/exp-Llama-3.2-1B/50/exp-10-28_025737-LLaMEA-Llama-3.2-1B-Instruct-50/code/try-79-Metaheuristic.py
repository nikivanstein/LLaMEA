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

class MutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func, mutation_rate):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        new_individual = [x for x in self.search_space if x not in best_func]
        new_individual = random.sample(new_individual, len(new_individual))
        new_individual = [x + random.uniform(-0.1, 0.1) for x in new_individual]
        new_individual = [x for x in new_individual if -5.0 <= x <= 5.0]
        new_individual = [x for x in new_individual if x not in best_func]

        # Apply mutation
        if random.random() < mutation_rate:
            new_individual = [x + random.uniform(-0.1, 0.1) for x in new_individual]
            new_individual = [x for x in new_individual if -5.0 <= x <= 5.0]

        # Update the search space
        self.search_space = new_individual

        return best_func, new_individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm uses mutation to refine the strategy, with a probability of 0.45 to apply mutation
# and a probability of 0.55 to not apply mutation