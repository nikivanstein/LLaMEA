import numpy as np
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

    def search(self, func, iterations=1000):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(iterations):
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

def BBOBMetaheuristicEvolutionaryAlgorithm(budget, dim, iterations=1000):
    # Create a new BBOBMetaheuristic instance
    algo = BBOBMetaheuristic(budget, dim)
    
    # Run the evolutionary algorithm
    best_solution = algo.search(func)
    
    # Return the best solution found
    return best_solution

def mutation_exp(budget, dim, iterations=1000):
    # Create a new BBOBMetaheuristic instance
    algo = BBOBMetaheuristic(budget, dim)
    
    # Run the evolutionary algorithm
    best_solution = algo.search(func)
    
    # Perform mutation
    for _ in range(iterations):
        # Randomly select a new individual
        new_individual = algo.search(func)
        
        # Check if the new individual is better than the current best
        if np.linalg.norm(new_individual - best_solution) < 1e-6:
            # Update the best solution
            best_solution = new_individual
    
    # Return the best solution found
    return best_solution

# BBOB test suite of 24 noiseless functions
def func(x):
    return x[0]**2 + x[1]**2

# Run the evolutionary algorithm
best_solution = BBOBMetaheuristicEvolutionaryAlgorithm(100, 2)

# Run the mutation-expansion algorithm
best_solution = mutation_exp(100, 2)