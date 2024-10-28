import numpy as np
import random

class ADELS:
    def __init__(self, budget, dim, mutation_rate=0.1, local_search_rate=0.1, local_search_threshold=5.0):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.local_search_rate = local_search_rate
        self.local_search_threshold = local_search_threshold
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

            # Perform local search
            if random.random() < self.local_search_rate and self.f < self.local_search_threshold:
                # Randomly select a direction for local search
                direction = np.random.uniform(-1, 1, self.dim)

                # Perform local search
                new_individual = self.x + direction * 0.1

                # Evaluate the new individual
                self.g = func(new_individual)

                # Check if the new individual is better
                if self.f < self.g:
                    # Update the current solution
                    self.x = new_individual
                    self.f = self.g

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

adecls = ADELS(1000, 10)
opt_x = adecls(__call__, test_func)
print(opt_x)