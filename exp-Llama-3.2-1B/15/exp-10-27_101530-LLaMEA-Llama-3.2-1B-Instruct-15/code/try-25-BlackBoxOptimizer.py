import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.refined_strategy = None

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
        # If the budget is reached, refine the strategy
        self.refine_strategy()
        # Return the best point found so far
        return self.search_space[0], self.search_space[1]

    def refine_strategy(self):
        if self.refined_strategy is None:
            # If no strategy has been refined, start with the original strategy
            self.refined_strategy = 'Novel Metaheuristic Algorithm for Black Box Optimization'
            return
        # Refine the strategy based on the current fitness
        if self.func_evaluations < 100:
            # If the number of function evaluations is low, use a simple strategy
            self.refined_strategy = 'Simple Strategy'
        elif self.func_evaluations < 500:
            # If the number of function evaluations is moderate, use a probabilistic strategy
            prob = 0.2
            self.refined_strategy = 'Probabilistic Strategy'
        else:
            # If the number of function evaluations is high, use a more advanced strategy
            prob = 0.3
            self.refined_strategy = 'Advanced Strategy'
        # Print the refined strategy
        print(f"Refined strategy: {self.refined_strategy}")

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 