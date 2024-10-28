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

class NovelMetaheuristicAlgorithm:
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

        # Apply mutation strategy
        mutated_func = self.mutate(best_func, num_evals)
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]
        mutated_func_values = [func(x) for x in random.sample(mutated_func, num_evals)]

        # Select the best mutated function value
        best_mutated_func = max(set(mutated_func_values), key=mutated_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_mutated_func]

        return best_mutated_func

    def mutate(self, individual, num_evals):
        # Generate a new individual by flipping the bits of the original individual
        mutated_individual = individual[:]
        for _ in range(num_evals // 2):
            idx = random.randint(0, len(mutated_individual) - 1)
            mutated_individual[idx] = 1 - mutated_individual[idx]
        return mutated_individual

# Initialize the algorithm
algorithm = NovelMetaheuristicAlgorithm(100, 10)

# Evaluate the function
def func(x):
    return np.sum(x**2)

best_func = algorithm(algorithm, func)
print("Best function:", best_func)
print("Score:", algorithm(best_func))