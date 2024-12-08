import numpy as np
import random
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

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual + np.random.normal(0, 0.1, self.dim)
        
        # Clip the mutated individual to the search space
        mutated_individual = np.clip(mutated_individual, bounds, None)
        
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.concatenate((parent1[:len(parent2)//2], parent2[len(parent2)//2:]))
        
        # Clip the child to the search space
        child = np.clip(child, bounds, None)
        
        return child

    def evolve(self, population_size, mutation_rate, crossover_rate):
        # Evolve the population
        population = [self.search(func) for func in self.search([np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)])]
        
        # Select the fittest individuals
        fittest_individuals = sorted(population, key=self.func_evals, reverse=True)[:self.budget]
        
        # Mutate and crossover the fittest individuals
        mutated_individuals = [self.mutate(individual) for individual in fittest_individuals]
        children = [self.crossover(parent1, parent2) for parent1, parent2 in zip(mutated_individuals, mutated_individuals)]
        
        # Replace the least fit individuals with the new ones
        population = [individual for individual in fittest_individuals if individual in children]
        
        # Replace the least fit individuals with new ones
        population = [individual for individual in population if individual not in children]
        
        return population