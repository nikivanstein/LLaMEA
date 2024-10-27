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

    def evolve(self, func):
        # Define the mutation strategy
        def mutate(individual):
            # Refine the strategy based on the probability 0.25
            if np.random.rand() < 0.25:
                # Randomly change a single element in the solution
                individual[np.random.randint(0, self.dim)] += np.random.uniform(-1, 1)
            return individual
        
        # Evolve the population for a specified number of generations
        population = [self.search(func) for _ in range(100)]
        
        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Select the fittest individuals
            fittest = sorted(population, key=self.func_evals, reverse=True)[:self.budget]
            
            # Create a new generation by mutating the fittest individuals
            new_generation = [mutate(individual) for individual in fittest]
            
            # Replace the old population with the new generation
            population = new_generation
        
        # Return the best individual from the new generation
        return self.search(func, population[0])

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 