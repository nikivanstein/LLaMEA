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
        new_individual = self.evaluate_fitness(best_func)
        self.search_space = [x for x in self.search_space if x not in new_individual]

        return best_func

    def evaluate_fitness(self, func):
        # Select the best individual from the search space
        best_individual = random.choice(self.search_space)

        # Evaluate the function using the best individual
        func_value = func(best_individual)

        # Update the search space with the new individual
        new_individual = self.search_space + [best_individual]

        # Apply mutation with probability 0.45
        if random.random() < 0.45:
            mutation = random.choice(new_individual)
            new_individual[new_individual.index(mutation)] = random.uniform(-5.0, 5.0)

        return new_individual

# Define the black box function
def func(x):
    return np.sum(x**2)

# Initialize the algorithm
algorithm = Metaheuristic(100, 10)

# Run the algorithm
best_func = algorithm(func)
print(f"Best function: {best_func}")
print(f"Best fitness: {func(best_func)}")