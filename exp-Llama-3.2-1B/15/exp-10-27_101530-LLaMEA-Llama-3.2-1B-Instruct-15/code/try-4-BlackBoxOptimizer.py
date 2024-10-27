import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
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
        # Refine the strategy by changing 20% of the individual lines
        for i in range(len(individual)):
            if random.random() < 0.2:
                individual[i] = random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def crossover(self, parent1, parent2):
        # Select two parents and create a new individual by combining them
        child = (parent1[:len(parent1)//2] + parent2[len(parent2)//2:])
        return child

    def selection(self, population):
        # Select the fittest individuals to reproduce
        return sorted(population, key=self.func_evaluations, reverse=True)[:self.budget//2]

# One-line description with main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses a novel combination of mutation, crossover, and selection to refine its strategy, allowing it to handle a wide range of tasks and optimize black box functions.

# Code: 