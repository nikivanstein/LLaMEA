import numpy as np
import random

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

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def mutate(self, individual):
        # Refine the strategy by changing the individual's lines
        # with a probability of 0.45
        if random.random() < 0.45:
            line = random.randint(0, self.dim - 1)
            individual[line] = random.uniform(-5.0, 5.0)

        return individual

# Initialize the algorithm
algorithm = NovelMetaheuristicAlgorithm(100, 10)

# Evaluate the function 100 times
func = lambda x: x**2
results = []
for _ in range(100):
    result = algorithm(func)
    results.append(result)

# Print the results
print("Optimized function:", results[-1])
print("Optimized function values:", results)