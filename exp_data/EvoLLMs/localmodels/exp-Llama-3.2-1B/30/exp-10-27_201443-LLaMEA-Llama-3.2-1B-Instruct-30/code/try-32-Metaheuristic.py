import numpy as np
from scipy.optimize import differential_evolution

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, initial_guess=None):
        if initial_guess is None:
            initial_guess = np.random.uniform(self.search_space)
        for _ in range(self.budget):
            func_value = func(initial_guess)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            initial_guess = np.random.uniform(self.search_space)
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

def optimize_function(func, initial_guess, budget=1000):
    metaheuristic = Metaheuristic(budget, funcdim=2)  # funcdim=2 for 2D space
    result = metaheuristic(initial_guess)
    return result

# Evaluate the selected solution
selected_solution = optimize_function(test_function, np.array([1.0, 1.0]))
print(selected_solution)