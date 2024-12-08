import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.budget_evaluations = 0
        self.history = []

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
        # Select two random points in the search space
        point1 = np.random.uniform(self.search_space[0], self.search_space[1])
        point2 = np.random.uniform(self.search_space[0], self.search_space[1])

        # Calculate the linear interpolation between the two points
        interpolation = (point2 - point1) / (self.search_space[1] - self.search_space[0])

        # Update the individual with the interpolated point
        individual = point1 + interpolation * (point2 - point1)

        # Ensure the individual remains within the search space
        individual = np.clip(individual, self.search_space[0], self.search_space[1])

        return individual

    def crossover(self, parent1, parent2):
        # Select a random point in the search space
        crossover_point = np.random.uniform(self.search_space[0], self.search_space[1])

        # Split the parents into two lists
        list1 = np.split(parent1, [crossover_point])
        list2 = np.split(parent2, [crossover_point])

        # Combine the lists to form a new individual
        child = np.concatenate((list1[0], list2[0], list1[1], list2[1]))

        # Return the child individual
        return child

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.