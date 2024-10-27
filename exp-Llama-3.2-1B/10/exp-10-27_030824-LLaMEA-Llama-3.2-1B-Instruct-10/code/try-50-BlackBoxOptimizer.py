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

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population = [BlackBoxOptimizer(budget, dim) for _ in range(100)]

    def __call__(self, func):
        # Select the fittest individual
        fittest_individual = max(self.population, key=lambda x: x.func_evaluations)
        # Generate a new individual by refining the fittest individual
        new_individual = fittest_individual
        for _ in range(self.dim):
            # Refine the new individual using a combination of random walk and linear interpolation
            new_point = new_individual + 0.1 * (np.random.uniform(-1, 1) * (fittest_individual.search_space[1] - fittest_individual.search_space[0]) + fittest_individual.search_space[0])
            new_individual = np.random.uniform(fittest_individual.search_space[0], fittest_individual.search_space[1])
            # Evaluate the new individual
            evaluation = func(new_point)
            # Update the new individual if the evaluation is better
            if evaluation > new_individual[1]:
                new_individual = new_point
        # Increment the function evaluations
        self.func_evaluations += 1
        # Return the new individual and its evaluation
        return new_individual, evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.