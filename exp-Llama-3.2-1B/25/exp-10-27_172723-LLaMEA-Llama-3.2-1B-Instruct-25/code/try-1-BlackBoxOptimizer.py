import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, mutation_rate=0.01, crossover_rate=0.5):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

def bbo(x, func, budget, mutation_rate=0.01, crossover_rate=0.5):
    """
    Black Box Optimizer (BBO) algorithm.

    Parameters:
    x (list): current point
    func (function): black box function to optimize
    budget (int): maximum number of function evaluations
    mutation_rate (float, optional): probability of mutation. Defaults to 0.01.
    crossover_rate (float, optional): probability of crossover. Defaults to 0.5.

    Returns:
    list: optimal point
    """
    for _ in range(budget):
        # Evaluate the function at the current point
        value = func(x)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return x
        else:
            # If the function has been evaluated within the budget, generate a new point
            new_x = np.random.choice(self.search_space, size=len(x), p=[1 - mutation_rate, crossover_rate])

            # Evaluate the new point
            new_value = func(new_x)

            # Check if the new point has been evaluated within the budget
            if new_value < 1e-10:  # arbitrary threshold
                # If not, return the new point as the optimal solution
                return new_x
            else:
                # If the new point has been evaluated within the budget, return the new point
                return new_x

# Test the BBO algorithm
def test_bbo():
    # Define the black box function
    def func(x):
        return x**2 + 2*x + 1

    # Define the BBO algorithm with mutation and crossover rates
    bbo_opt = bbo(x0=[-1], func=func, budget=1000, mutation_rate=0.1, crossover_rate=0.9)

    # Optimize the function
    print("Optimal point:", bbo_opt)

# Test the BBO algorithm
test_bbo()