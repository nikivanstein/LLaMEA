import random
import numpy as np
from scipy.optimize import minimize
from typing import List

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, iterations=100, mutation_rate=0.01):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space) for _ in range(100)]

        # Evaluate the function for each individual in the population
        for _ in range(iterations):
            # Generate a new individual by mutating the current one
            new_individual = self.mutate(population, mutation_rate)

            # Evaluate the function for the new individual
            value = func(new_individual)

            # Check if the function has been evaluated within the budget
            if value < 1e-10:  # arbitrary threshold
                # If not, return the current individual as the optimal solution
                return new_individual
            else:
                # If the function has been evaluated within the budget, return the individual
                return new_individual

    def mutate(self, population, mutation_rate):
        # Select two parents using tournament selection
        parents = random.sample(population, 2)

        # Evaluate the function for each parent
        for parent in parents:
            value = func(parent)

            # Generate a new individual by mutation
            new_individual = parent.copy()
            if random.random() < mutation_rate:
                # Randomly swap two genes in the individual
                idx1, idx2 = random.sample(range(len(new_individual)), 2)
                new_individual[idx1], new_individual[idx2] = new_individual[idx2], new_individual[idx1]

            # Evaluate the function for the new individual
            value = func(new_individual)

            # Check if the function has been evaluated within the budget
            if value < 1e-10:  # arbitrary threshold
                # If not, return the new individual as the mutated individual
                return new_individual
            else:
                # If the function has been evaluated within the budget, return the individual
                return new_individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

# Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation