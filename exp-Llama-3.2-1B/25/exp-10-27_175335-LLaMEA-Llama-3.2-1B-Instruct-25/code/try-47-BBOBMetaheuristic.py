import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func, bounds, mutation_rate):
        # Define the search space
        bounds = np.linspace(bounds[0], bounds[1], self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Apply mutation to the solution
        mutated_sol = sol.copy()
        for _ in range(int(self.budget * mutation_rate)):
            # Randomly select an individual from the population
            individual = mutated_sol
            
            # Randomly select a mutation point
            mutation_point = random.randint(0, self.dim - 1)
            
            # Apply mutation
            mutated_sol[mutation_point] = random.uniform(bounds[0], bounds[1])
        
        # Return the best solution found
        return sol

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# BBOBMetaheuristic: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# 
# def __init__(self, budget, dim):
#     self.budget = budget
#     self.dim = dim
#     self.func_evals = 0

def optimize_function(func, bounds, mutation_rate, budget):
    return BBOBMetaheuristic(budget, bounds).search(func, bounds, mutation_rate)

# Example usage:
bounds = [0, 10]
mutation_rate = 0.1
budget = 100
func = lambda x: x**2

best_solution = optimize_function(func, bounds, mutation_rate, budget)
print("Best solution:", best_solution)