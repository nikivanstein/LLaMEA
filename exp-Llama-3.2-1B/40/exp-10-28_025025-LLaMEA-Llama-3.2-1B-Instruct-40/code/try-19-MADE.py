# Description: Refinement-based MADE algorithm for black box optimization
# Code: 
# ```python
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
        self.refine_history = []

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

            # Refine the solution using probability 0.4
            self.refine_solution()

    def refine_solution(self):
        # Select a random individual from the history
        self.x = np.random.choice(self.x_history, size=self.dim, replace=False)

        # Evaluate the new individual using the budget
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

    def evaluateBBOB(self, func, num_evals):
        # Evaluate the function at the current solution using the given number of evaluations
        for _ in range(num_evals):
            self.f = func(self.x)