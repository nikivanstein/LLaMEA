import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.search_space_copy = deepcopy(self.search_space)

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def select_strategy(self, individual):
        # Refine the strategy based on the fitness value and the number of evaluations
        if self.func_evaluations == 0:
            return individual

        # Calculate the average fitness value
        avg_fitness = np.mean([self.func(individual, self.logger) for individual in self.search_space_copy])

        # Calculate the strategy score based on the average fitness value and the number of evaluations
        strategy_score = 0.2 * avg_fitness + 0.8 * np.sum([self.func(individual, self.logger) for individual in self.search_space_copy])

        # Refine the strategy based on the strategy score
        if strategy_score > 0.5:
            # Increase the search space to increase the number of evaluations
            self.search_space_copy = np.linspace(-5.0, 5.0, 200)
            return self.select_strategy(deepcopy(individual))
        elif strategy_score < -0.5:
            # Decrease the search space to decrease the number of evaluations
            self.search_space_copy = np.linspace(5.0, 5.0, 100)
            return self.select_strategy(deepcopy(individual))
        else:
            # Use the current strategy
            return individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Refine the strategy
optimizer = BBOBBlackBoxOptimizer(1000, 10)
optimizer = optimizer.select_strategy(optimizer.select_strategy(optimizer.select_strategy(optimizer.select_strategy(optimizer.select_strategy(optimizer.select_strategy(optimizer.select_strategy(deepcopy(func), self.logger))))))
print(result)