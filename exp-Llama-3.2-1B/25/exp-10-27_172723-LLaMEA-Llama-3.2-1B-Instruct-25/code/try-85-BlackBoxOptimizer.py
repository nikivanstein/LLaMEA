import random
import numpy as np

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

    def __next_solution(self, population, mutation_rate):
        # Select the fittest individual from the population
        fittest_individual = population[np.argmax([x[1] for x in population])]

        # Create a new individual by refining the fittest individual
        new_individual = fittest_individual
        for _ in range(random.randint(1, self.dim)):
            # Randomly select a dimension to refine
            dimension = random.randint(0, self.dim - 1)

            # Refine the individual by adding a random value to the selected dimension
            new_individual[dimension] += random.uniform(-1, 1)

        # Check if the new individual has been evaluated within the budget
        if np.any([x[1] for x in [new_individual]] + [f(x) for f in population]) < 1e-10:
            # If not, return the new individual as the optimal solution
            return new_individual
        else:
            # If the new individual has been evaluated within the budget, return the new individual
            return new_individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"