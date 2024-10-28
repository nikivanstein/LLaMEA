import numpy as np
import random

class DEXBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []
        self.crossover_rate = 0.7
        self.mutation_rate = 0.01

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Insufficient budget")

        # Initialize the current solution
        self.x = np.random.uniform(-5.0, 5.0, self.dim)
        self.f = func(self.x)

        # Initialize the mutation rate
        self.m = 0.1

        while self.budget > 0:
            # Evaluate the function at the current solution
            self.f = func(self.x)

            # Generate two new solutions using differential evolution
            self.g1 = self.x + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)
            self.g1 = np.clip(self.g1, -5.0, 5.0)

            self.g2 = self.x + np.random.normal(0, 1, self.dim) * np.sqrt((self.f - self.g1) / self.budget)
            self.g2 = np.clip(self.g2, -5.0, 5.0)

            # Evaluate the new solutions
            self.g1 = func(self.g1)
            self.g2 = func(self.g2)

            # Check if the new solutions are better
            if self.f < max(self.g1, self.g2):
                # Update the current solution
                self.x = self.g1
                self.f = self.g1

                # Update the mutation rate
                self.m = 0.1

            # Update the history
            self.x_history.append(self.x)
            self.m_history.append(self.m)

            # Decrease the budget
            self.budget -= 1

            # Check if the budget is zero
            if self.budget == 0:
                break

        return self.x

# One-line description: Differential Evolution with Evolutionary Crossover for Black Box Optimization
# Code: 
# ```python
# DEXBO: Differential Evolution with Evolutionary Crossover for Black Box Optimization
# ```
# ```python
# def test_func(x):
#     return np.sum(x**2)

# dexbbo = DEXBO(1000, 10)
# opt_x = dexbbo(__call__, test_func)
# print(opt_x)