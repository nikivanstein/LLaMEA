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

        # Refine the strategy
        if random.random() < 0.45:
            self.search_space = self._refine_search_space(self.search_space)

        return best_func

    def _refine_search_space(self, search_space):
        # Select the next individual based on the current search space
        next_individual = random.sample(search_space, 1)[0]

        # Select the next mutation based on the current search space
        mutation_prob = 0.1
        if random.random() < mutation_prob:
            next_individual[0] = random.uniform(-5.0, 5.0)
            if random.random() < 0.5:
                next_individual[1] = random.uniform(-5.0, 5.0)

        return next_individual

# Initialize the metaheuristic
metaheuristic = Metaheuristic(10, 10)

# Evaluate the function 10 times
func = lambda x: x**2
best_func = metaheuristic(func)
print(best_func)