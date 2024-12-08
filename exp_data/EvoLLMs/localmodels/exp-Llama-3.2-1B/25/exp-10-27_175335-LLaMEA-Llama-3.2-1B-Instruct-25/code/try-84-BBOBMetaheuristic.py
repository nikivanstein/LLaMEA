import numpy as np
import random
import copy
import os

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
                sol = copy.deepcopy(sol)
        
        # Return the best solution found
        return sol

class EvolutionaryMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the population
        self.population = self.search(func)
        
        # Define the mutation strategy
        def mutate(individual):
            # Change the individual lines of the selected solution to refine its strategy
            lines = individual.tolist()
            if random.random() < 0.25:
                # Refine the strategy by changing the individual lines
                lines[random.randint(0, len(lines) - 1)] = f"Refine the strategy by changing the individual lines to {random.randint(1, len(lines))}"
            return ''.join(lines)
        
        # Mutate the population
        self.population = [mutate(individual) for individual in self.population]
        
        # Evaluate the function at the best individual
        best_individual = self.population[0]
        best_func_sol = self.__call__(func, best_individual)
        
        # Return the best individual found
        return best_individual, best_func_sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 