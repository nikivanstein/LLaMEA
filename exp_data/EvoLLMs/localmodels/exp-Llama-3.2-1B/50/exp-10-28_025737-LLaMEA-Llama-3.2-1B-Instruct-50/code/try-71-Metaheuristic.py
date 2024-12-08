# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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

    def mutate(self, individual):
        # Refine the strategy by changing the individual lines
        if random.random() < 0.45:
            # Change the individual lines to refine the strategy
            return [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
        else:
            # Return the individual as it is
            return individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_values = [func(x) for x in individual]
        return max(set(func_values), key=func_values.count)

# Initialize the algorithm
algorithm = Metaheuristic(100, 10)

# Run the algorithm
best_individual = algorithm(10)
best_func = algorithm(best_individual)

# Print the result
print("Best Individual:", best_individual)
print("Best Function:", best_func)