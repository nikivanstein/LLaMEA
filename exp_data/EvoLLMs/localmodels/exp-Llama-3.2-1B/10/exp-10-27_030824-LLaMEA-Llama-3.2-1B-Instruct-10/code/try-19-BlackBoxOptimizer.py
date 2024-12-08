# Description: Novel Metaheuristic Algorithm for Black Box Optimization using a Novel Combination of Random Walk and Linear Interpolation
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.current_point = None
        self.current_evaluation = None
        self.population = []

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            self.current_point = np.random.uniform(self.search_space[0], self.search_space[1])
            self.current_evaluation = func(self.current_point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Store the point and evaluation
            self.population.append((self.current_point, self.current_evaluation))
            # Return the point and evaluation
            return self.current_point, self.current_evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            if len(self.population) > 0:
                return self.population[0][0], self.population[0][1]
            else:
                return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Randomly select an individual from the population
        i = random.randint(0, len(self.population) - 1)
        # Swap the current point with the selected individual
        self.population[i] = (self.population[i][0], self.population[i][1])

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# New code
class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.current_point = None
        self.current_evaluation = None
        self.population = []

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            self.current_point = np.random.uniform(self.search_space[0], self.search_space[1])
            self.current_evaluation = func(self.current_point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Store the point and evaluation
            self.population.append((self.current_point, self.current_evaluation))
            # Return the point and evaluation
            return self.current_point, self.current_evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            if len(self.population) > 0:
                return self.population[0][0], self.population[0][1]
            else:
                return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Randomly select two individuals from the population
        i = random.randint(0, len(self.population) - 1)
        j = random.randint(0, len(self.population) - 1)
        # Swap the current point with the selected individual
        self.population[i] = (self.population[i][0], self.population[i][1])
        self.population[j] = (self.population[j][0], self.population[j][1])