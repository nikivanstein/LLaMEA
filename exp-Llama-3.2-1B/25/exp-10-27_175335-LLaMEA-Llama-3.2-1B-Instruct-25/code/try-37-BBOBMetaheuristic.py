import numpy as np
from scipy.optimize import differential_evolution
from collections import deque

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.population_size = 100
        self.mutation_rate = 0.1

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
        
        # Update the best solution found
        self.best_solution = sol
        self.best_fitness = func(sol)

        # Return the best solution found
        return sol

    def mutate(self, individual):
        # Generate a new individual with refined strategy
        new_individual = individual.copy()
        
        # Change the strategy by changing the direction of the gradient
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                new_individual[i] += np.random.uniform(-1, 1)
        
        # Ensure the new individual stays within the bounds
        new_individual = np.clip(new_individual, -5.0, 5.0)
        
        return new_individual

    def evolve(self):
        # Evolve the population using differential evolution
        population = deque([self.search(func) for _ in range(self.population_size)])
        
        # Evolve for a specified number of generations
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.best_fitness, reverse=True)[:self.population_size//2]
            
            # Mutate the fittest individuals
            mutated_individuals = [self.mutate(individual) for individual in fittest_individuals]
            
            # Replace the least fit individuals with the mutated ones
            population = deque([individual for individual in fittest_individuals if individual in mutated_individuals])
            
            # Replace the least fit individuals with the new ones
            population = deque([individual for individual in fittest_individuals if individual not in mutated_individuals])
        
        # Return the best solution found
        return population[0]

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Refine the strategy of the solution by changing the direction of the gradient
# through mutation