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

        return self.x

    def mutate(self, individual):
        # Refine the individual based on a probability of refinement
        if random.random() < 0.4:
            # Generate a new mutation direction
            direction = np.random.normal(0, 1, self.dim)

            # Refine the individual using differential evolution
            self.x += direction * np.sqrt(self.f / self.budget)
            self.x = np.clip(self.x, -5.0, 5.0)

    def anneal(self, initial_solution, cooling_rate):
        # Simulate simulated annealing
        current_solution = initial_solution
        temperature = 1.0
        while temperature > 0.01:
            # Generate a new solution using simulated annealing
            new_solution = current_solution + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / temperature)

            # Evaluate the new solution
            new_solution = func(new_solution)

            # Check if the new solution is better
            if new_solution < self.f:
                # Update the current solution
                current_solution = new_solution

            # Decrease the temperature using cooling rate
            temperature *= cooling_rate

        return current_solution

# Example usage:
def test_func(x):
    return np.sum(x**2)

made = MADE(1000, 10)
opt_x = made(__call__, test_func)

# Refine the initial solution
for _ in range(100):
    made.mutate(made.x)

# Anneal the solution using simulated annealing
initial_solution = made.x
for _ in range(100):
    made.anneal(initial_solution, 0.99)

print(opt_x)