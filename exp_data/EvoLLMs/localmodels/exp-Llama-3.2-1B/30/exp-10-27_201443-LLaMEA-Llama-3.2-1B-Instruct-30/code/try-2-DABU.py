import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def adaptive_search(self, func, initial_solution, alpha=0.1, beta=0.01):
        # Generate a random initial solution
        solution = initial_solution
        
        # Perform adaptive search
        while self.func_evaluations < self.budget:
            # Evaluate the function at the current solution
            func_value = func(solution)
            
            # If the function value is close to zero, stop
            if np.abs(func_value) < 1e-6:
                break
            
            # Update the solution using the best-fitting data point
            solution = np.array([x for x, _ in zip(self.search_space, func_value)])
            
            # Update the search space using the best-fitting data point
            best_idx = np.argmax(np.abs(func_value))
            self.search_space[best_idx] = alpha * self.search_space[best_idx] + beta * (1 - alpha) * (1 - best_idx)
        
        # Refine the solution using the best-fitting data point
        best_idx = np.argmax(np.abs(func_value))
        self.search_space[best_idx] = alpha * self.search_space[best_idx] + beta * (1 - alpha) * (1 - best_idx)
        
        return solution

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

def best_func(x):
    return np.exp(-x[0]**2 - x[1]**2)

best_solution = dabu.adaptive_search(best_func, x0=[1, 1])
print(best_solution)  # prints a refined solution

def random_func(x):
    return np.random.rand(2)

random_solution = dabu.adaptive_search(random_func, x0=[1, 1])
print(random_solution)  # prints a random solution