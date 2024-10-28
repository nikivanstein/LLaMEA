import random
import numpy as np
from scipy.optimize import minimize

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
        # Refine the strategy by changing 0.45 of the individual's lines
        mutated_individual = individual.copy()
        for _ in range(int(self.budget * 0.45)):
            idx = random.randint(0, self.dim - 1)
            mutated_individual[idx] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def fitness(self, individual):
        # Evaluate the fitness of the individual
        func_values = [individual[i] for i in range(self.dim)]
        return np.mean(func_values)

# Initialize the algorithm
algorithm = Metaheuristic(100, 10)