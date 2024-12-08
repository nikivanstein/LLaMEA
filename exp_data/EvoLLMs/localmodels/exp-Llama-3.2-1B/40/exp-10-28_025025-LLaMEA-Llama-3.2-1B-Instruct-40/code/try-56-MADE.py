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
        self.evolutionary_strategies = {
            'uniform': self.uniform_evolutionary_strategy,
            'bounded': self.bounded_evolutionary_strategy
        }

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

        # Apply evolutionary strategy to refine the solution
        strategy = random.choice(self.evolutionary_strategies)
        if strategy == 'uniform':
            strategy = self.uniform_evolutionary_strategy
        elif strategy == 'bounded':
            strategy = self.bounded_evolutionary_strategy

        # Refine the solution using the chosen strategy
        self.x = strategy(self.x, self.f, self.m, self.budget)

        return self.x

    def uniform_evolutionary_strategy(self, x, f, m, budget):
        # Generate a new solution using uniform mutation
        new_x = x + np.random.normal(0, 1, self.dim) * np.sqrt(f / budget)
        new_x = np.clip(new_x, -5.0, 5.0)

        # Evaluate the new solution
        new_x = func(new_x)

        # Check if the new solution is better
        if new_x < f:
            # Update the current solution
            self.x = new_x
            self.f = new_x

            # Update the mutation rate
            self.m = m

        # Update the history
        self.x_history.append(x)
        self.m_history.append(m)

        # Decrease the budget
        self.budget -= 1

        # Check if the budget is zero
        if self.budget == 0:
            break

        return self.x

    def bounded_evolutionary_strategy(self, x, f, m, budget):
        # Generate a new solution using bounded mutation
        new_x = x + np.random.normal(0, 1, self.dim) * np.sqrt(f / budget)
        new_x = np.clip(new_x, -5.0, 5.0)

        # Evaluate the new solution
        new_x = func(new_x)

        # Check if the new solution is better
        if new_x < f:
            # Update the current solution
            self.x = new_x
            self.f = new_x

            # Update the mutation rate
            self.m = m

        # Update the history
        self.x_history.append(x)
        self.m_history.append(m)

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

# Description: Differential Evolution with Evolutionary Strategies
# Code: 