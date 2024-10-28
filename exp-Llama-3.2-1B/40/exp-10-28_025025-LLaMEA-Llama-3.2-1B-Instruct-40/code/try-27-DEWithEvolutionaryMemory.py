import numpy as np
import random

class DEWithEvolutionaryMemory:
    def __init__(self, budget, dim, mutation_rate=0.1, evolutionary_memory_size=100):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.evolutionary_memory = np.random.uniform(-5.0, 5.0, self.dim)
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
        self.m = self.mutation_rate

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
                self.m = np.clip(self.m + random.uniform(-self.mutation_rate, self.mutation_rate), 0, 1)

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

de_with_evolutionary_memory = DEWithEvolutionaryMemory(1000, 10)
opt_x = de_with_evolutionary_memory(__call__, test_func)
print(opt_x)

# Refine the solution using evolutionary memory
def refine_solution(x, evolutionary_memory):
    # Evaluate the function at the current solution
    f = test_func(x)
    # Generate a new solution using differential evolution
    g = x + np.random.normal(0, 1, evolutionary_memory.shape)
    g = np.clip(g, -5.0, 5.0)
    # Evaluate the new solution
    g = test_func(g)
    # Check if the new solution is better
    if f < g:
        # Update the current solution
        x = g
        f = g
    # Update the evolutionary memory
    evolutionary_memory = np.clip(evolutionary_memory + random.uniform(-evolutionary_memory.shape[0]*0.1, evolutionary_memory.shape[0]*0.1), 0, evolutionary_memory.shape[0])
    return x, evolutionary_memory

refined_x, refined_evolutionary_memory = refine_solution(opt_x, de_with_evolutionary_memory.evolutionary_memory)
print(refined_x)
print(refined_evolutionary_memory)