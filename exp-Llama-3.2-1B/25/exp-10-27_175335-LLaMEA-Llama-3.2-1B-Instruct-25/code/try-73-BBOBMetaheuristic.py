import numpy as np
from scipy.optimize import differential_evolution
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

def bbob_metaheuristic(budget, dim, func):
    """
    Evolutionary Algorithm for Black Box Optimization using Genetic Programming.
    
    Args:
    budget (int): The number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    func (function): The black box function to optimize.
    
    Returns:
    function: The optimized function.
    """
    # Initialize the population with random solutions
    population = [random.uniform(-5.0, 5.0) for _ in range(100)]
    
    # Run the evolution for the specified number of generations
    for _ in range(100):
        # Evaluate the fitness of each individual
        fitness = [self.__call__(func, individual) for individual in population]
        
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[::-1][:budget]
        
        # Create a new population by mutating the fittest individuals
        new_population = [individual + random.uniform(-0.1, 0.1) for individual in fittest_individuals]
        
        # Replace the old population with the new one
        population = new_population
    
    # Return the fittest individual in the final population
    return population[0]

# Define the test function
def test_function(x):
    return np.sum(x)

# Create a new instance of the BBOBMetaheuristic algorithm
bbob_metaheuristic_instance = BBOBMetaheuristic(100, 10, test_function)

# Optimize the test function using the algorithm
optimized_function = bbob_metaheuristic_instance(bbob_metaheuristic_instance.budget, bbob_metaheuristic_instance.dim, test_function)

# Save the optimized function to a file
np.save("optimized_function.npy", optimized_function)