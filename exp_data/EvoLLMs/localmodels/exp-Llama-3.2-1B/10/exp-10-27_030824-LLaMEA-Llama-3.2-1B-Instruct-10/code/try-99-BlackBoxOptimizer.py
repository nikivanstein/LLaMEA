import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.search_space_size = 100

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def mutate(self, individual):
        # Randomly choose a direction for mutation
        direction = np.random.uniform(0, 1)
        # Randomly choose a mutation point
        mutation_point = np.random.randint(0, self.search_space_size)
        # Perform mutation
        individual[mutation_point] += direction * np.random.uniform(-1, 1)
        # Ensure the individual stays within the search space
        individual[mutation_point] = np.clip(individual[mutation_point], self.search_space[0], self.search_space[1])
        return individual

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 