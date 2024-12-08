import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, adaptive_strategy):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.adaptive_strategy = adaptive_strategy

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

    def mutate(self):
        # Update the adaptive strategy based on the evaluation history
        if self.func_evaluations < self.budget // 4:
            self.adaptive_strategy = "line_search"
        elif self.func_evaluations < self.budget // 2:
            self.adaptive_strategy = "random_search"
        else:
            self.adaptive_strategy = "adaptive_search"

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = self.evaluate_fitness(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

# One-line description: "Adaptive Black Box Optimizer: A novel metaheuristic algorithm that efficiently adapts its search strategy to handle a wide range of optimization tasks"

# Budget: 1000
# Dimension: 5
# Adaptive strategy: random_search