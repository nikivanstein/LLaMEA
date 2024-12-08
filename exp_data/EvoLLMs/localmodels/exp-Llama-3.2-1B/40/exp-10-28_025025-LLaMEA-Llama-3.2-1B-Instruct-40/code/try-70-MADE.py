import numpy as np
import random
from scipy.optimize import differential_evolution

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

            # Generate a new solution using differential evolution
            res = differential_evolution(lambda x: -x, [(self.x + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget), self.x)], bounds=((-5.0, 5.0), (-5.0, 5.0)))
            self.g = res.x[0]

            # Evaluate the new solution
            self.g = func(self.g)

            # Check if the new solution is better
            if self.f < self.g:
                # Update the current solution
                self.x = self.g
                self.f = self.g

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

# Example usage:
def test_func(x):
    return np.sum(x**2)

made = MADE(1000, 10)
opt_x = made(__call__, test_func)
print(opt_x)

# Novel Heuristic Algorithm: Evolutionary Differential Evolution (EDE)
# Description: EDE combines the strengths of differential evolution and evolutionary algorithms.
# Code: 