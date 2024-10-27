import random
import numpy as np

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

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def __repr__(self):
        return self.__str__()

    def mutate(self, individual):
        # Refine the strategy by changing the individual lines
        # 15% probability of changing a line
        if random.random() < 0.15:
            # Randomly select a line
            line_index = random.randint(0, self.dim - 1)
            # Randomly select a new value for the line
            new_value = random.uniform(-5.0, 5.0)
            # Replace the line with the new value
            individual[line_index] = new_value
        return individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 