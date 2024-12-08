import numpy as np
import random
import math

class AEoO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []

    def __call__(self, func, logger=None):
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

            # Update the history
            self.x_history.append(self.x)
            self.m_history.append(self.m)

            # Decrease the budget
            self.budget -= 1

            # Check if the budget is zero
            if self.budget == 0:
                break

        # Refine the strategy using adaptive techniques
        if logger is not None:
            # Use the probability 0.4 to change the individual lines of the selected solution
            for i in range(len(self.x_history)):
                if random.random() < 0.4:
                    self.x[i] += np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)
                    self.f = func(self.x[i])

            # Update the mutation rate
            self.m = 0.1 / len(self.x_history)

        return self.x

# Example usage:
def test_func(x):
    return np.sum(x**2)

aeo_o = AEoO(1000, 10)
opt_x = aeo_o(__call__, logger=None)
print(opt_x)
