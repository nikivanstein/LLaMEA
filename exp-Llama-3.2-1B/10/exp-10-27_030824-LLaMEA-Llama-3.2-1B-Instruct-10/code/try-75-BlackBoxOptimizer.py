import random
import numpy as np
import copy
from collections import deque

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

    def generate_initial_population(self, population_size):
        # Generate a population of random individuals
        population = [copy.deepcopy(func(np.random.uniform(self.search_space[0], self.search_space[1])) for func in self.funcs) for _ in range(population_size)]
        return population

    def mutate(self, individual):
        # Randomly swap two random elements in the individual
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual

    def linear_interpolate(self, individual1, individual2):
        # Interpolate between two individuals using linear interpolation
        return [self.linear_interpolate(individual1, individual2, i) for i in range(len(individual1))]

    def random_walk(self, individual):
        # Perform a random walk from the individual to the search space
        walk = []
        while walk[-1] < self.search_space[1]:
            walk.append(random.uniform(self.search_space[0], self.search_space[1]))
        walk.append(self.search_space[1])
        return walk

    def __next__(self):
        # Select the next individual based on the probability of each strategy
        if self.func_evaluations < self.budget:
            # Random walk strategy
            if random.random() < 0.5:
                return self.random_walk(np.random.uniform(self.search_space[0], self.search_space[1]))
            # Linear interpolation strategy
            else:
                return self.linear_interpolate(np.random.uniform(self.search_space[0], self.search_space[1]), np.random.uniform(self.search_space[0], self.search_space[1]))
        else:
            # Return a default individual
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.