import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('-inf')

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

    def mutate(self, individual):
        if self.best_individual is None:
            # If this is the first iteration, select the best individual so far
            self.best_individual = individual
            self.best_fitness = self.evaluate_fitness(individual)
        else:
            # Otherwise, select a random individual from the current best
            best_individual = random.choice([self.best_individual, individual])
            # Calculate the fitness of the best individual
            fitness = self.evaluate_fitness(best_individual)
            # Mutate the best individual with a probability based on its fitness
            if random.random() < 0.1 * fitness / self.best_fitness:
                best_individual = random.choice([best_individual, individual])
            # Return the mutated individual
            return best_individual

    def evaluate_fitness(self, individual):
        # Calculate the fitness of the individual
        fitness = np.array([func(individual, self.logger) for func in self.funcs])
        # Return the fitness
        return fitness

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 