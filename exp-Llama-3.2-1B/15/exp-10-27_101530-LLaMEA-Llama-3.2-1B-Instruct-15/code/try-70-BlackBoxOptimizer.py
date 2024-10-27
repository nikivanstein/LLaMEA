import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        # Refine the strategy by changing the individual lines of the selected solution
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Mutate the individual by changing one line of the strategy
        if random.random() < 0.15:
            # Change the individual lines of the strategy
            individual[0] = random.uniform(self.search_space[0], self.search_space[1])
            individual[1] = random.uniform(self.search_space[0], self.search_space[1])

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual using the Black Box function
        func_value = self.func(individual)
        # Return the fitness of the individual
        return func_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 