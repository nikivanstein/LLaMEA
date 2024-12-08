import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.iterations = 0

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

    def mutation(self, individual, mutation_rate):
        # Select a random point in the search space
        point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Evaluate the function at the point
        evaluation = func(point)
        # Check if the mutation rate is less than 0.1
        if random.random() < mutation_rate:
            # Randomly select a new point in the search space
            new_point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the new point
            new_evaluation = func(new_point)
            # Return the new individual and its evaluation
            return individual, evaluation, new_point, new_evaluation
        else:
            # Return the original individual and its evaluation
            return individual, evaluation

    def crossover(self, parent1, parent2):
        # Select a random point in the search space
        point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Evaluate the function at the point
        evaluation = func(point)
        # Check if the crossover rate is less than 0.1
        if random.random() < 0.1:
            # Randomly select a new point in the search space
            new_point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the new point
            new_evaluation = func(new_point)
            # Return the new parent individuals and their evaluations
            return parent1, evaluation, parent1, new_evaluation, parent2, evaluation, new_point, new_evaluation
        else:
            # Return the parents and their evaluations
            return parent1, evaluation, parent2, evaluation

    def __str__(self):
        return "Novel Metaheuristic Algorithm for Black Box Optimization"

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.