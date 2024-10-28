# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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
        new_individual = copy.deepcopy(best_func)
        new_individual = np.array(new_individual)
        for _ in range(self.dim):
            new_individual = new_individual / 2.0
            if np.random.rand() < 0.45:
                new_individual = new_individual + np.random.uniform(-0.5, 0.5)
        self.search_space = [x for x in self.search_space if x not in new_individual]

        return best_func, new_individual

# Initialize the algorithm
algorithm = Metaheuristic(100, 10)

# Test the algorithm
def test_function():
    return np.random.uniform(-5.0, 5.0)

# Evaluate the function
func = test_function
best_func, new_individual = algorithm(func)

# Print the result
print(f"Best function: {best_func}")
print(f"New individual: {new_individual}")