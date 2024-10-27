import random
import numpy as np
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
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

        # Flip the bit at the selected index
        individual[idx] = 1 - individual[idx]

        # If the individual is a string, convert it to a list
        if isinstance(individual, str):
            individual = [individual]

        # If the individual is a list, convert it to a string
        elif isinstance(individual, list):
            individual = ''.join(individual)

        # If the individual is a numpy array, convert it to a list
        elif isinstance(individual, np.ndarray):
            individual = individual.tolist()

        return individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"