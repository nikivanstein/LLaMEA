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
        # Refine the strategy by changing the probability of selection
        individual = self.evaluate_fitness(individual)
        new_probabilities = [random.random() for _ in range(self.dim)]
        new_probabilities = [p / sum(new_probabilities) for p in new_probabilities]
        new_individual = [x * (1 - p) + p * y for x, y in zip(individual, new_probabilities)]
        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual using the budget
        func_values = [func(x) for x in individual]
        return np.mean(func_values)

# Initialize the algorithm with the budget and dimension
metaheuristic = Metaheuristic(100, 5)

# Evaluate the function 100 times
func = lambda x: x**2
results = [metaheuristic(individual) for individual in range(100)]

# Print the results
print("Results:")
for i, result in enumerate(results):
    print(f"Individual {i+1}: {result} with fitness {metaheuristic.evaluate_fitness(result)}")

# Update the search space
metaheuristic.search_space = [x for x in metaheuristic.search_space if x not in results[0]]

# Print the updated search space
print("\nUpdated Search Space:")
for x in metaheuristic.search_space:
    print(x)

# Optimize the function using the updated search space
new_individual = metaheuristic.mutate(results[0])

# Evaluate the function using the new individual
new_fitness = metaheuristic.evaluate_fitness(new_individual)

# Print the new fitness
print(f"\nNew Fitness: {new_fitness}")

# Print the updated individual
print(f"\nUpdated Individual: {new_individual}")