import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_point = None
        self.best_value = -np.inf
        self.current_best_point = None
        self.current_best_value = -np.inf

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return self.best_value

    def evaluate_fitness(self, individual, logger):
        updated_individual = self.f(individual, logger)
        self.func_evals += 1
        if updated_individual > self.current_best_point:
            self.current_best_point = updated_individual
            self.current_best_value = self.evaluate_fitness(updated_individual, logger)
        return self.current_best_value

    def f(self, individual, logger):
        # Generate a random point in the search space
        point = np.random.uniform(-5.0, 5.0, self.dim)
        # Evaluate the function at the point
        value = func(point)
        # Check if the point is within the bounds
        if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
            # If the point is within bounds, update the function value
            value = self.evaluate_fitness(individual, logger)
        # Change the individual line of the selected solution to refine its strategy
        # self.func_evals += 1
        # self.func_evals = min(self.func_evals, self.budget)
        # self.best_point = np.max(self.func_evals * np.random.uniform(-5.0, 5.0, self.dim))
        # self.best_value = np.max(self.func_evals * np.random.uniform(-5.0, 5.0, self.dim))
        # self.current_best_point = np.max(self.func_evals * np.random.uniform(-5.0, 5.0, self.dim))
        # self.current_best_value = np.max(self.func_evals * np.random.uniform(-5.0, 5.0, self.dim))
        return value