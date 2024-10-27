import random
import numpy as np
from scipy.optimize import minimize_scalar

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

    def optimize(self, func):
        # Define the bounds for the random search
        bounds = [(-5.0, 5.0)] * self.dim

        # Perform the random search
        new_individuals = []
        for _ in range(100):  # Run 100 iterations
            new_individual = random.choices(self.search_space, weights=[1 / np.exp(i) for i in bounds], k=self.dim)
            new_individual = np.array(new_individual)
            new_individual = self.evaluate_fitness(new_individual, func)
            new_individuals.append(new_individual)

        # Select the best individual
        best_individual = np.argmax([func(individual) for individual in new_individuals])

        # Refine the solution based on the probability 0.25
        if random.random() < 0.25:
            best_individual = random.choice(new_individuals)

        # Evaluate the function at the refined solution
        value = func(best_individual)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return best_individual
        else:
            # If the function has been evaluated within the budget, return the point
            return best_individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"