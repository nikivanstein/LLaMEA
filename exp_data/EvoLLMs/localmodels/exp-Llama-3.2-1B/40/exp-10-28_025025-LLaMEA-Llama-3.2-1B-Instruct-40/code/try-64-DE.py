import numpy as np
import random

class DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []
        self.m_rate = 0.4

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
                self.m = max(0, min(1, self.m * self.m_rate))

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

de = DE(1000, 10)
opt_x = de(__call__, test_func)
print(opt_x)

# Refine the solution with adaptive mutation rate
def refine_de(de, opt_x, func, budget):
    de.m_rate = 0.7
    de.budget = budget
    de.x = opt_x
    de.f = func(de.x)
    de.m = 0.1

    while de.budget > 0:
        de.x = de.g
        de.f = func(de.x)
        de.m = max(0, min(1, de.m * de.m_rate))
        de.budget -= 1
        de.m = 0.1
        if de.budget == 0:
            break
        de.g = func(de.x)
    return de.x

de_refined = refine_de(de, opt_x, test_func, 1000)
print(de_refined)

# Description: Differential Evolution with Adaptive Mutation Rate
# Code: 
# ```python
# Refine the solution with adaptive mutation rate
# ```
# ```python
# def refine_de(de, opt_x, func, budget):
#     de.m_rate = 0.7
#     de.budget = budget
#     de.x = opt_x
#     de.f = func(de.x)
#     de.m = 0.1

#     while de.budget > 0:
#         de.x = de.g
#         de.f = func(de.x)
#         de.m = max(0, min(1, de.m * de.m_rate))
#         de.budget -= 1
#         de.m = 0.1
#         if de.budget == 0:
#             break
#         de.g = func(de.x)
#     return de.x