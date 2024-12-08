import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"DABU: {self.budget} evaluations, {self.dim} dimensions"

    def __str_func_evals__(self, func_evals):
        return f"DABU: {func_evals} evaluations"

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Novel Metaheuristic Algorithm: Adaptive Multi-Step Hill Climbing (AMSHC)
# The algorithm adapts its strategy based on the convergence rate of the current solution
class AMSHC:
    def __init__(self, budget, dim, step_size, alpha, beta):
        self.budget = budget
        self.dim = dim
        self.step_size = step_size
        self.alpha = alpha
        self.beta = beta
        self.func_evaluations = 0
        self.current_solution = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            if self.current_solution is None:
                self.current_solution = func(self.search_space)
                self.func_evaluations += 1
            else:
                # Calculate the new solution using the current solution and step size
                new_solution = self.current_solution + self.step_size * np.random.normal(0, 1, self.dim)
                new_func_value = func(new_solution)

                # Evaluate the new function value
                if np.abs(new_func_value) < 1e-6:  # stop if the function value is close to zero
                    break

                # Update the current solution and function evaluations
                self.current_solution = new_solution
                self.func_evaluations += 1

                # Adapt the strategy based on the convergence rate
                if new_func_value < self.alpha * self.func_evaluations:
                    # Increase the step size to accelerate convergence
                    self.step_size *= self.beta
                elif new_func_value > self.alpha * self.func_evaluations + self.beta * self.func_evaluations:
                    # Decrease the step size to slow down convergence
                    self.step_size /= self.beta

    def __str__(self):
        return f"AMSHC: {self.budget} evaluations, {self.dim} dimensions"

    def __str_func_evals__(self, func_evals):
        return f"AMSHC: {func_evals} evaluations"

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

amshc = AMSHC(1000, 2, 0.1, 0.5, 1.0)  # 1000 function evaluations, 2 dimensions
print(amshc(test_function))  # prints a random value between -10 and 10