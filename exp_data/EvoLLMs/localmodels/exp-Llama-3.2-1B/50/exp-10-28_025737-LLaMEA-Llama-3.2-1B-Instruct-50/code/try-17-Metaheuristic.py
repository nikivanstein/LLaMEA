import numpy as np
import random
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

        # Refine the strategy
        if len(self.search_space) < 10:
            self.search_space = [x for x in self.search_space if x not in best_func]
            self.search_space = random.sample(self.search_space, len(self.search_space))
        elif len(self.search_space) < 20:
            self.search_space = [x for x in self.search_space if x not in best_func]
            self.search_space = random.sample(self.search_space, len(self.search_space) // 2)
        elif len(self.search_space) < 50:
            self.search_space = [x for x in self.search_space if x not in best_func]
            self.search_space = random.sample(self.search_space, len(self.search_space) // 5)

        return best_func

class MutationExp:
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

        # Refine the strategy
        if len(self.search_space) < 10:
            new_individual = self.search_space[0]
            for i in range(self.dim):
                if random.random() < 0.1:
                    new_individual[i] += np.random.uniform(-0.1, 0.1)
            self.search_space = [x for x in self.search_space if x not in new_individual]
        elif len(self.search_space) < 20:
            new_individual = self.search_space[0]
            for i in range(self.dim):
                if random.random() < 0.1:
                    new_individual[i] += np.random.uniform(-0.2, 0.2)
            self.search_space = [x for x in self.search_space if x not in new_individual]
        elif len(self.search_space) < 50:
            new_individual = self.search_space[0]
            for i in range(self.dim):
                if random.random() < 0.1:
                    new_individual[i] += np.random.uniform(-0.3, 0.3)
            self.search_space = [x for x in self.search_space if x not in new_individual]

        # Perform mutation
        mutated_individual = new_individual[:]

        # Select a mutation probability
        mutation_prob = 0.45

        # Apply mutation
        for i in range(self.dim):
            if random.random() < mutation_prob:
                mutated_individual[i] += np.random.uniform(-0.1, 0.1)

        # Evaluate the function
        func_values = [func(mutated_individual[i]) for i in range(self.dim)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

# Initialize the metaheuristic algorithm
metaheuristic = Metaheuristic(100, 10)

# Evaluate the function
def func(x):
    return x[0]**2 + x[1]**2

best_func = metaheuristic(func)

# Print the results
print("Best function:", best_func)
print("Best fitness:", metaheuristic.evaluate_fitness(best_func))