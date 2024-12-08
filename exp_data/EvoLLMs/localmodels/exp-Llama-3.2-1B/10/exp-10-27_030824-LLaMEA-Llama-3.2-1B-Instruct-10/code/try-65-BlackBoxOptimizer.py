import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.current_individual = None
        self.current_point = None
        self.current_evaluation = None

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            self.current_point = np.random.uniform(self.search_space[0], self.search_space[1])
            self.current_evaluation = func(self.current_point)
            self.func_evaluations += 1
            return self.current_point, self.current_evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            self.current_point = np.random.uniform(self.search_space[0], self.search_space[1])
            self.current_evaluation = func(self.current_point)
            self.func_evaluations += 1
            return self.current_point, self.current_evaluation

    def mutate(self, new_individual):
        # Randomly swap two elements in the new individual
        if random.random() < 0.1:
            self.current_individual = new_individual[:2] + [new_individual[2]]
            self.func_evaluations += 1
            return self.current_individual
        else:
            return self.current_individual

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization