import numpy as np
from scipy.optimize import minimize
from collections import deque

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = deque(maxlen=1000)
        self.pop_size = 50
        self.mutation_rate = 0.1
        self.exploration_rate = 0.45
        self.step_size = 1.0

    def __call__(self, func):
        # Initialize the population with random solutions
        solutions = np.random.uniform(-5.0, 5.0, size=self.pop_size)
        
        # Evaluate the function for each solution
        evaluations = [minimize(func, solution, args=(solutions,)) for solution in solutions]
        
        # Select the best solutions based on the budget
        selected_solutions = np.array([solution for _, solution in evaluations[:self.budget]])
        
        # Update the population with the selected solutions
        self.population = deque(selected_solutions)
        
        # Perform exploration-exploitation trade-off
        if np.random.rand() < self.exploration_rate:
            # Explore the search space by increasing the step size
            self.step_size *= 1.1
        else:
            # Exploit the current best solutions by decreasing the step size
            self.step_size /= 1.1
        
        # Return the best solution found so far
        return np.array(self.population[0])

# Define a test function
def test_func(x):
    return x[0]**2 + x[1]**2

# Create an instance of the evolutionary algorithm
ea = EvolutionaryAlgorithm(budget=100, dim=2)

# Evaluate the test function 100 times
for _ in range(100):
    func = test_func
    best_func = ea(func)
    best_func_score = -ea(best_func)
    print(f"Best function: {best_func}, Score: {best_func_score}")
    print(f"Best solution: {best_func}, Dimension: {ea.dim}")
    print(f"Exploration rate: {ea.exploration_rate}")
    print(f"Step size: {ea.step_size}")
    print("------------------------")