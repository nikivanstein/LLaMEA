import numpy as np
import random

class DifferentialEvolutionWithMutation:
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

# Example usage:
def test_func(x):
    return np.sum(x**2)

def differential_evolution_with_mutation(func, budget, dim):
    made = DifferentialEvolutionWithMutation(budget, dim)
    opt_x = made(func)
    return opt_x

def mutation_exp(func, budget, dim, mutation_rate, mutation_strength):
    return differential_evolution_with_mutation(func, budget, dim)

# Evaluate the black box function
def evaluate_bboB(func, budget, dim):
    return mutation_exp(func, budget, dim, 0.4, 0.1)

# Test the algorithm
func = test_func
budget = 1000
dim = 10

opt_x = evaluate_bboB(func, budget, dim)
print(opt_x)

# Refine the solution using probability 0.4
def refine_solution(opt_x, func, budget, dim):
    mutation_rate = 0.4
    mutation_strength = 0.1
    new_individual = opt_x
    new_individual = new_individual + np.random.normal(0, 1, dim) * np.sqrt(func(new_individual) / budget)
    new_individual = np.clip(new_individual, -5.0, 5.0)
    new_individual = evaluate_bboB(func, budget, dim)
    new_individual = opt_x + (new_individual - opt_x) * mutation_rate
    return new_individual

refined_x = refine_solution(opt_x, func, budget, dim)
print(refined_x)