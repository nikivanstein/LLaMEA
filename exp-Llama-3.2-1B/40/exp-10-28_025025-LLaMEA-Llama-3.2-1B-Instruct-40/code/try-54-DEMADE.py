import numpy as np
import random

class DEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = []
        self.m_history_best = []
        self.m_history_best_score = -np.inf

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
                self.m_history_best.append(self.m)
                self.m_history_best_score.append(self.m_history_best[-1])

                # Check if the budget is zero
                if self.budget == 0:
                    break

            # Update the mutation rate
            self.m *= 0.9 + 0.1 * np.random.normal(0, 1)

            # Decrease the budget
            self.budget -= 1

            # Check if the budget is zero
            if self.budget == 0:
                break

        return self.x

# Example usage:
def test_func(x):
    return np.sum(x**2)

de = DEMADE(1000, 10)
opt_x = de(__call__, test_func)
print(opt_x)

# Refine the strategy
def refine_strategy(individual):
    # Evaluate the function at the current solution
    self.f = func(individual)

    # Generate a new solution using differential evolution
    self.g = individual + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)
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
        self.m_history_best.append(self.m)
        self.m_history_best_score.append(self.m_history_best[-1])

        # Check if the budget is zero
        if self.budget == 0:
            break

    return self.x

# Apply the refinement strategy
refined_x = refine_strategy(opt_x)
print(refined_x)

# Update the algorithm with the refined strategy
def update_algorithm(algorithm, refined_x):
    algorithm.m = 0.1
    algorithm.m_history = []
    algorithm.x_history = []
    algorithm.m_history_best = []
    algorithm.m_history_best_score = -np.inf
    algorithm.f = refined_x
    return algorithm

# Apply the updated algorithm
updated_algorithm = update_algorithm(de, refined_x)
opt_x = updated_algorithm(__call__, test_func)
print(opt_x)

# Evaluate the updated algorithm
updated_f = test_func(updated_x)
print(updated_f)

# Refine the strategy again
refined_strategy_2 = refine_strategy(updated_x)
opt_x = refined_strategy_2
print(opt_x)

# Evaluate the updated algorithm
updated_f = test_func(updated_x)
print(updated_f)