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

        # Refine the strategy by changing the individual lines
        self.refine_strategy()

        return best_func

    def refine_strategy(self):
        # Calculate the probability of each individual line
        probabilities = [1 / len(self.search_space) for _ in range(self.dim)]

        # Select the next individual line based on the probabilities
        next_individual = random.choices(range(self.dim), weights=probabilities)[0]

        # Update the search space with the selected individual line
        self.search_space = [x for x in self.search_space if x!= next_individual]

        # Refine the strategy by changing the individual lines
        self.refine_strategy()

    def mutate(self, individual):
        # Select a random individual line
        line_index = random.randint(0, self.dim - 1)
        # Change the individual line to a random line
        self.search_space[line_index] = random.uniform(-5.0, 5.0)

        # Refine the strategy by changing the individual lines
        self.refine_strategy()

# Initialize the Metaheuristic algorithm
metaheuristic = Metaheuristic(100, 10)

# Evaluate the function using the Metaheuristic algorithm
def func(x):
    return x[0]**2 + x[1]**2

best_func = metaheuristic(100, 2)
print(best_func)