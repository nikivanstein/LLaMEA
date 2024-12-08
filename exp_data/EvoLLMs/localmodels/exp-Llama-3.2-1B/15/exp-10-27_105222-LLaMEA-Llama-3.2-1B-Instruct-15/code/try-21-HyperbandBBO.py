# Description: Efficient Black Box Optimization using Hyperband and Bayesian Optimization
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, dim):
        # Define the Bayesian optimization algorithm
        def bayesian_optimization(func, dim, budget):
            # Initialize the population
            population = np.random.uniform(*self.search_space, size=(budget, dim))
            
            # Define the hyperband search space
            hyperband_search_space = (np.min(self.search_space), np.max(self.search_space))
            
            # Initialize the best individual and its fitness
            best_individual = None
            best_fitness = float('-inf')
            
            # Perform Bayesian optimization
            for _ in range(budget):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*hyperband_search_space, size=dim)
                
                # Evaluate the function at the new point
                func_value = func(x)
                
                # Update the best individual and its fitness
                if func_value > best_fitness:
                    best_individual = x
                    best_fitness = func_value
            
            # Return the best individual
            return best_individual
        
        # Define the hyperband search
        def hyperband_search(func, dim, budget):
            # Initialize the population
            population = np.random.uniform(*self.search_space, size=(budget, dim))
            
            # Define the hyperband search space
            hyperband_search_space = (np.min(self.search_space), np.max(self.search_space))
            
            # Initialize the best individual and its fitness
            best_individual = None
            best_fitness = float('-inf')
            
            # Perform hyperband search
            for i in range(budget):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*hyperband_search_space, size=dim)
                
                # Evaluate the function at the new point
                func_value = func(x)
                
                # Update the best individual and its fitness
                if func_value > best_fitness:
                    best_individual = x
                    best_fitness = func_value
            
            # Return the best individual
            return best_individual
        
        # Define the Bayesian optimization algorithm
        def optimize_func(func, dim):
            # Initialize the population
            population = np.random.uniform(*self.search_space, size=(self.budget, dim))
            
            # Define the hyperband search
            hyperband_search_func = hyperband_search(func, dim, self.budget)
            
            # Define the Bayesian optimization algorithm
            def bayesian_optimization_func(population, dim):
                # Optimize the function using the Bayesian optimization algorithm
                return minimize(bayesian_optimization, func, method="SLSQP", bounds=[(self.search_space[0], self.search_space[1])], constraints={"type": "eq", "fun": lambda x: x[0] - x[1]})
            
            # Optimize the function using the Bayesian optimization algorithm
            return bayesian_optimization_func(population, dim)
        
        # Optimize the function using the Bayesian optimization algorithm
        return optimize_func(func, dim)

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()