import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
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
        
        # Return the best solution found
        return sol

    def mutate(self, func, sol):
        # Refine the solution by changing one line of the strategy
        new_sol = sol.copy()
        line_index = random.randint(0, self.dim - 1)
        new_sol[line_index] = random.uniform(-5.0, 5.0)
        
        # Evaluate the new solution
        new_func_sol = self.__call__(func, new_sol)
        
        # Check if the new solution is better than the current best
        if new_func_sol < self.__call__(func, sol):
            # Update the solution
            sol = new_sol
        
        return sol, new_func_sol

    def evolve(self, func, num_generations):
        # Initialize the population
        population = [self.search(func) for _ in range(100)]
        
        # Evolve the population
        for generation in range(num_generations):
            # Mutate the population
            population = [self.mutate(func, sol) for sol in population]
            
            # Evaluate the fittest individuals
            fittest_sol = max(population, key=lambda sol: self.__call__(func, sol))
            
            # Return the fittest individual
            return fittest_sol, self.__call__(func, fittest_sol)

# One-line description with the main idea
# BBOBMetaheuristic: An evolutionary algorithm for black box optimization using genetic programming, which can handle a wide range of tasks and is evaluated on the BBOB test suite of 24 noiseless functions.
# Code: 