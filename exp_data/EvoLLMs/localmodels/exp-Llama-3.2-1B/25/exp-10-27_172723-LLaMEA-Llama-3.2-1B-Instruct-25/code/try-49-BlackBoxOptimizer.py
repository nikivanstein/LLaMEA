import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def select_strategy(self, func, budget):
        # Define the strategy to select the next point
        if self.budget == 0:
            # If the budget is 0, select a random point from the search space
            return np.random.choice(self.search_space)
        else:
            # If the budget is not 0, select a point that minimizes the function value
            return np.argmin(func(np.random.choice(self.search_space, size=self.dim)))

    def mutate(self, func, budget):
        # Define the mutation function to introduce new points
        def mutate_point(point):
            # Generate a new point by perturbing the current point
            new_point = point + np.random.normal(0, 0.1, self.dim)
            # Evaluate the new point
            value = func(new_point)
            # Check if the new point is within the budget
            if value < 1e-10:
                # If not, return the new point as the mutated point
                return new_point
            else:
                # If the new point is within the budget, return the point
                return point

        # Evaluate the function at the current point
        value = func(self.select_strategy(func, budget))

        # Check if the function has been evaluated within the budget
        if value < 1e-10:
            # If not, return the current point as the mutated point
            return self.mutate_point(self.select_strategy(func, budget))
        else:
            # If the function has been evaluated within the budget, return the point
            return self.mutate_point(self.select_strategy(func, budget))

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
