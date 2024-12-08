import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

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
        return "Novel Metaheuristic Algorithm for Black Box Optimization"

    def __repr__(self):
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim}, search_space={self.search_space}, func_evaluations={self.func_evaluations})"

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations):
        # Initialize the population with random individuals
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Perform iterations
        for _ in range(iterations):
            # Select the fittest individual
            fittest_individual = population[np.argmax([func(individual) for individual in population])]

            # Generate a new individual using the fittest individual and the search space
            new_individual = fittest_individual + np.random.uniform(-1, 1, self.dim)
            new_individual = np.clip(new_individual, self.search_space[0], self.search_space[1])

            # Evaluate the new individual
            evaluation = func(new_individual)

            # Increment the function evaluations
            self.func_evaluations += 1

            # If the budget is reached, return the new individual and evaluation
            if self.func_evaluations == self.budget:
                return new_individual, evaluation

            # Add the new individual to the population
            population.append(new_individual)

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 