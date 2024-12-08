import random
import numpy as np
import operator

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
        new_search_space = [x for x in self.search_space if x not in best_func]

        # Evaluate the new search space
        new_num_evals = min(self.budget, len(self.evaluate_fitness(new_search_space)))

        # Select the best function value in the new search space
        new_best_func = max(set(self.evaluate_fitness(new_search_space)), key=self.evaluate_fitness)

        # Update the search space
        self.search_space = new_search_space

        return new_best_func

    def evaluate_fitness(self, func):
        return np.mean([func(x) for x in random.sample(self.search_space, 10)])

    def mutate(self, func):
        # Refine the strategy by changing the probability of selecting each individual line
        probabilities = [0.1, 0.3, 0.6]  # 10% chance of each individual line
        new_individual = random.choices(self.search_space, weights=probabilities, k=10)
        return new_individual

# Test the algorithm
func = lambda x: x**2
metaheuristic = Metaheuristic(100, 5)
print(metaheuristic(metaheuristic(func)))

# Refine the strategy
metaheuristic = Metaheuristic(100, 5)
print(metaheuristic(metaheuristic.metaheuristic(func)))

# Mutate the algorithm
metaheuristic = Metaheuristic(100, 5)
print(metaheuristic(metaheuristic.metaheuristic(metaheuristic(func))))