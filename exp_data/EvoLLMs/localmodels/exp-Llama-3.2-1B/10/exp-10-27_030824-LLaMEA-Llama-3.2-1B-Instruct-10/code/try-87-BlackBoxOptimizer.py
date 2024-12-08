import random
import numpy as np
from scipy.optimize import minimize
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, num_iterations):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Evaluate the function for each point in the population
        for _ in range(num_iterations):
            for individual in population:
                evaluation = func(individual)
                # Increment the function evaluations
                self.func_evaluations += 1
                # Return the point and its evaluation
                return individual, evaluation

        # If the population is exhausted, return a default point and evaluation
        return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population = deque(maxlen=100)

    def __call__(self, func, num_iterations):
        # Initialize the population with random points in the search space
        for _ in range(num_iterations):
            individual, evaluation = self.evaluate_fitness(func)
            self.population.append(individual)
            # Return the point and its evaluation
            return individual, evaluation

    def evaluate_fitness(self, func):
        # Generate a random point in the search space
        point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Evaluate the function at the point
        evaluation = func(point)
        # Increment the function evaluations
        self.func_evaluations += 1
        # Return the point and its evaluation
        return point, evaluation

    def mutate(self, individual):
        # Randomly swap two points in the individual
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        # Return the mutated individual
        return individual

# Example usage:
optimizer = NovelMetaheuristic(100, 5)
func = lambda x: np.sin(x)
num_iterations = 1000
individual, evaluation = optimizer(func, num_iterations)
print(f"Fitness: {evaluation}")