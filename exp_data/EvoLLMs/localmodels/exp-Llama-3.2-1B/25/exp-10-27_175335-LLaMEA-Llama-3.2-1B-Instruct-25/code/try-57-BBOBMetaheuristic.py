import numpy as np
import random
import copy

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
            sol = copy.deepcopy(bounds)
            
            # Refine the solution using probability 0.25 to change the individual lines of the solution
            for _ in range(self.dim):
                if random.random() < 0.25:
                    sol[_] = random.uniform(-5.0, 5.0)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

class GeneticProgramming(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function within the budget
        func_evals = 0
        for individual in self.population:
            func_evals += self.func_evals
            self.func_evals += 1
            func(individual, func)
        
        # Return the best solution found
        return max(self.population, key=self.func_evals)

    def mutate(self, individual):
        # Refine the individual using probability 0.25 to change the individual lines of the solution
        for _ in range(self.dim):
            if random.random() < 0.25:
                individual[_] = random.uniform(-5.0, 5.0)
        
        # Return the mutated individual
        return individual

    def evaluate_fitness(self, func, individual):
        # Evaluate the function at the individual
        func(individual, func)

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Refines the solution by changing individual lines of the solution using probability 0.25
# 
# Code: 