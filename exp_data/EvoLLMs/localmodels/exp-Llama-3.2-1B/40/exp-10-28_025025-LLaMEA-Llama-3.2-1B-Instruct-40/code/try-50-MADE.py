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
        self.refining_strategy = None

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

        # Refine the solution using the refining strategy
        self.refining_strategy()
        return self.x

    def refining_strategy(self):
        # Generate a random candidate solution
        candidate = self.x + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)

        # Evaluate the candidate solution
        candidate = func(candidate)

        # Check if the candidate solution is better
        if candidate < self.f:
            # Update the current solution
            self.x = candidate
            self.f = candidate

            # Update the mutation rate
            self.m = 0.1

        # Update the history
        self.x_history.append(self.x)
        self.m_history.append(self.m)

# Example usage:
def test_func(x):
    return np.sum(x**2)

made = MADE(1000, 10)
opt_x = made(__call__, test_func)
print(opt_x)

# Refining strategy
class RefiningStrategy:
    def __init__(self, func, bounds, mutation_rate):
        self.func = func
        self.bounds = bounds
        self.mutation_rate = mutation_rate

    def __call__(self, individual):
        # Generate a random candidate solution
        candidate = individual + np.random.normal(0, 1, self.bounds[1] - self.bounds[0]) * self.mutation_rate

        # Evaluate the candidate solution
        candidate = self.func(candidate)

        # Check if the candidate solution is better
        if candidate < self.func(individual):
            # Update the current solution
            individual = candidate
            self.func(individual)

            # Update the mutation rate
            self.mutation_rate = 0.1

        return individual

refining_strategy = RefiningStrategy(test_func, (-5.0, 5.0), 0.1)
made.refining_strategy = refining_strategy
opt_x = made(__call__, test_func)
print(opt_x)