import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.budgets = np.logspace(-2, 0, 100).astype(int)  # Initialize a list of possible budgets

    def __call__(self, func):
        best_point = None
        best_score = -np.inf
        for budget in self.budgets:
            new_individual = self.evaluate_fitness(func, budget)
            if new_individual is not None:
                updated_individual = self.f(new_individual, self.logger)
                if updated_individual is not None and updated_individual['score'] > best_score:
                    best_point = updated_individual['point']
                    best_score = updated_individual['score']
        return best_point, best_score

    def f(self, new_individual, logger):
        func_value = new_individual['func'](new_individual['point'])
        if logger is not None:
            logger.update(func_value)
        return {'point': new_individual['point'], 'func': new_individual['func'],'score': func_value}

    def evaluate_fitness(self, func, budget):
        if budget == 0:
            return None
        # Generate a random point in the search space
        point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        # Evaluate the function at the point
        func_value = func(point)
        # Increment the function evaluations
        self.func_evaluations += 1
        # Check if the point is within the budget
        if self.func_evaluations < budget:
            # If not, return the point
            return point
        else:
            # If the budget is reached, return the best point found so far
            return self.search_space[0], self.search_space[1]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 