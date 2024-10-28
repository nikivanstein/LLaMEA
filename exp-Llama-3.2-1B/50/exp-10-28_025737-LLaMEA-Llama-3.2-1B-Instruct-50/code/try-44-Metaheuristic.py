import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def mutate(self, individual):
        # Refine the strategy by changing the probability of mutation
        if random.random() < 0.45:
            # Change the lower bound
            self.search_space[0] = np.random.uniform(0.0, 5.0)
            # Change the upper bound
            self.search_space[1] = np.random.uniform(0.0, 5.0)
        else:
            # Change the upper bound
            self.search_space[1] = np.random.uniform(0.0, 5.0)
            # Change the lower bound
            self.search_space[0] = np.random.uniform(0.0, 5.0)

# Initialize the algorithm
algorithm = Metaheuristic(100, 10)