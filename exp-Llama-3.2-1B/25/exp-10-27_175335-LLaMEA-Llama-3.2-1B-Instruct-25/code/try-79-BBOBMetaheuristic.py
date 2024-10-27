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

    def search(self, func, mutation_rate):
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
        for _ in range(self.budget // 2):
            # Randomly select an individual
            individual = sol
            
            # Randomly select a mutation point
            mutation_point = random.randint(0, self.dim - 1)
            
            # Apply mutation
            mutated_individual = individual.copy()
            mutated_individual[mutation_point] += random.uniform(-1, 1)
            
            # Update the solution
            sol = mutated_individual
        
        # Return the best solution found
        return sol

class BBOBMetaheuristicEvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.func = None
        self.population = []

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func is None:
            self.func = func
            self.population = []
        
        # Evaluate the function within the budget
        func_evals = self.func_evals = 0
        while func_evals < self.budget:
            # Search for a solution
            sol = self.search(func, self.mutation_rate)
            
            # Add the solution to the population
            self.population.append(sol)
            
            # Evaluate the function at the solution
            func_evals += 1
        
        # Return the best solution found
        return self.population[np.argmax([self.func(sol) for sol in self.population])]

# Example usage
budget = 100
dim = 10
mutation_rate = 0.01
algorithm = BBOBMetaheuristicEvolutionaryAlgorithm(budget, dim, mutation_rate)
best_solution = algorithm(__call__)

# Save the best solution
np.save("currentexp/aucs-BBOBMetaheuristicEvolutionaryAlgorithm-0.npy", best_solution)