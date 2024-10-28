import numpy as np
import random
import math

class MADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []
        self.m_history_best = []
        self.x_history_best = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Insufficient budget")

        # Initialize the current solution
        self.x = np.random.uniform(-5.0, 5.0, self.dim)
        self.f = func(self.x)

        # Initialize the mutation rate
        self.m = 0.1

        # Initialize the mutation strategy
        self.m_strategy = np.random.uniform(0.5, 1.5)

        while self.budget > 0:
            # Evaluate the function at the current solution
            self.f = func(self.x)

            # Generate a new solution using differential evolution
            self.g = self.x + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)
            self.g = np.clip(self.g, -5.0, 5.0)

            # Evaluate the new solution
            self.g = func(self.g)

            # Check if the new solution is better
            if self.f < self.g:
                # Update the current solution
                self.x = self.g
                self.f = self.g

                # Update the mutation rate
                self.m = 0.1

                # Update the mutation strategy
                self.m_strategy = np.random.uniform(0.5, 1.5)

            # Update the history
            self.x_history.append(self.x)
            self.m_history.append(self.m)
            self.m_history_best.append(self.m_history_best[-1])
            self.x_history_best.append(self.x_history_best[-1])

            # Decrease the budget
            self.budget -= 1

            # Check if the budget is zero
            if self.budget == 0:
                break

        # Update the mutation strategy based on the best fitness found so far
        self.m_strategy = min(self.m_strategy, 0.5)
        self.m_strategy = max(self.m_strategy, 1.5)

        return self.x

# Example usage:
def test_func(x):
    return np.sum(x**2)

made = MADE(1000, 10)
opt_x = made(__call__, test_func)
print(opt_x)

# One-line description with the main idea:
# Differential Evolution with Adaptive Mutation Strategy
# This algorithm combines differential evolution and adaptive mutation strategy to optimize black box functions.