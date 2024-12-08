import numpy as np
from scipy.optimize import minimize
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

    def optimize(self, func, initial_solution, budget):
        # Initialize the population
        population = [initial_solution]
        
        # Run the evolutionary algorithm for the specified budget
        for _ in range(budget):
            # Evaluate the function at each individual in the population
            fitnesses = [self.func_evals]
            for individual in population:
                fitness = self.func_evals
                # Evaluate the function at the individual
                func_evals = self.func_evals
                func_evals += 1
                fitnesses.append(fitness)
            
            # Select the fittest individuals
            indices = np.argsort(fitnesses)[:len(population)//2]
            population = [population[i] for i in indices]
            
            # Perform crossover and mutation
            for i in range(len(population)):
                parent1 = population[i]
                parent2 = population[(i+1)%len(population)]
                child = (parent1 + parent2) / 2
                if random.random() < 0.5:
                    # Perform crossover
                    child[:self.dim//2] = parent1[:self.dim//2]
                    child[self.dim//2: self.dim] = parent2[self.dim//2:]
                else:
                    # Perform mutation
                    child[random.randint(0, self.dim-1)] += random.uniform(-1, 1)
                
                # Update the population
                population[i] = child
        
        # Return the best individual found
        return population[0]