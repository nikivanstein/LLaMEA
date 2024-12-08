import numpy as np
import random
import copy

class DES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []
        self.nbs = None
        self.nbs_history = []

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

        # Apply stochastic neighborhood search
        self.nbs = np.random.uniform(-5.0, 5.0, self.dim)
        self.nbs = np.clip(self.nbs, -5.0, 5.0)
        self.nbs_history.append(self.nbs)

        # Refine the solution using the best of the neighborhood
        self.x = self.x + 0.3 * (self.nbs - self.x)

        return self.x

# Example usage:
def test_func(x):
    return np.sum(x**2)

des = DES(1000, 10)
opt_x = des(__call__, test_func)
print(opt_x)

# One-line description with the main idea:
# DES is a novel metaheuristic algorithm that combines differential evolution with stochastic neighborhood search, allowing for efficient and effective optimization of black box functions.