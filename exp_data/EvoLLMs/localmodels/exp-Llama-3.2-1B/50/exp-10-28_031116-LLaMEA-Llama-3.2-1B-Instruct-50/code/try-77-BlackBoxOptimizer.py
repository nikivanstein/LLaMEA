import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population = np.random.uniform(self.search_space, size=(self.budget, dim))

    def __call__(self, func):
        def optimize_func(x):
            return func(x)
        
        # Initialize the population with random solutions
        self.population = np.random.uniform(self.search_space, size=(self.budget, self.dim))
        
        # Evaluate the function for each solution in the population
        for i in range(self.budget):
            # Refine the search space using probability 0.45
            self.search_space = np.concatenate((self.search_space[:int(0.45 * self.budget)], self.search_space[int(0.45 * self.budget):]))
            
            # Select the best solution based on the budget
            best_solution = np.argmax(optimize_func(self.population[i]))
            best_x = self.population[i][best_solution]
            
            # Update the population
            self.population[i] = np.random.uniform(self.search_space, size=(self.dim,))
        
        # Evaluate the final function for the best solution
        best_func_value = optimize_func(best_x)
        
        # Update the selected solution
        best_solution = np.argmax(optimize_func(self.population))
        best_x = self.population[best_solution]
        
        # Store the updated solution
        self.best_solution = best_x
        self.best_func_value = best_func_value
        
        return best_solution, best_func_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 