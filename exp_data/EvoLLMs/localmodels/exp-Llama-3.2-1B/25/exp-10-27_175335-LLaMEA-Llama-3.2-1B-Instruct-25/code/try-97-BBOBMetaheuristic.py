import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func, mutation_rate=0.01, mutation_threshold=0.1):
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
            
            # Apply mutation to the solution
            if random.random() < mutation_rate:
                # Randomly select an index in the search space
                idx = random.randint(0, self.dim-1)
                
                # Randomly change the value at the selected index
                sol[idx] = random.uniform(bounds[idx])
        
        # Update the best solution and fitness
        self.best_individual = sol
        self.best_fitness = func(sol)

        # Check if the best solution has a better fitness than the current best
        if self.best_fitness < self.best_fitness:
            self.best_fitness = self.best_fitness

        # Return the best solution found
        return sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
def fitness(individual):
    func = BBOBMetaheuristic(1000, 10).search(lambda x: x**2)
    return x - func

# Initialize the BBOBMetaheuristic with a budget of 1000 function evaluations
bpm = BBOBMetaheuristic(1000, 10)

# Evaluate the function for 1000 times
for _ in range(1000):
    func = bpm.search(lambda x: x**2)
    bpm.func_evals += 1

# Print the best solution found
print("Best individual:", bpm.best_individual)
print("Best fitness:", bpm.best_fitness)