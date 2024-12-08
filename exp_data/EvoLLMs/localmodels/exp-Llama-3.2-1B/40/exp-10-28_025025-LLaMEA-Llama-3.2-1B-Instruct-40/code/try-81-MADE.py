import numpy as np
import random

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
        self.p_history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Insufficient budget")

        # Initialize the current solution
        self.x = np.random.uniform(-5.0, 5.0, self.dim)
        self.f = func(self.x)

        # Initialize the mutation rate
        self.m = 0.1

        # Initialize the probability of mutation
        self.p = 0.4

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

                # Update the probability of mutation
                self.p = 0.4

            # Update the history
            self.x_history.append(self.x)
            self.m_history.append(self.m)
            self.p_history.append(self.p)

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

# Adapt the solution to refine its strategy
def adapt_solution(x, p):
    # Update the mutation rate based on the probability of mutation
    self.m = max(0.01, min(0.1, self.m * p))

    # Update the probability of mutation
    self.p = max(0.01, min(0.9, self.p * (1 - p)))

# Example usage:
def adapt_made(x):
    adapt_solution(x, 0.7)

adapt_made(opt_x)
print(opt_x)