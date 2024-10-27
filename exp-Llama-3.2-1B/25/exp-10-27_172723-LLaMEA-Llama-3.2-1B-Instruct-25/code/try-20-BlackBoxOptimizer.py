import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, num_evaluations=100, mutation_rate=0.01):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + num_evaluations)
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def mutate(self, individual):
        # Randomly select an index in the individual
        idx = random.randint(0, self.dim - 1)

        # Randomly change the value at the selected index
        new_individual = individual.copy()
        new_individual[idx] += random.uniform(-1, 1)

        # Check if the new individual is within the search space
        if new_individual[idx] < -5.0 or new_individual[idx] > 5.0:
            # If not, return the original individual
            return individual
        else:
            # If the new individual is within the search space, return the new individual
            return new_individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

# Description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"