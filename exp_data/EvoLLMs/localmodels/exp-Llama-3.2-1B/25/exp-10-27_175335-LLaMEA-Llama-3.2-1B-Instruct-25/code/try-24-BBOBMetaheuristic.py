import numpy as np
import random
import copy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []
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

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = copy.deepcopy(bounds)
            
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
        mutated_individual = copy.deepcopy(individual)
        mutated_individual[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        
        return mutated_individual

    def evolve(self):
        # Evolve the population
        while self.func_evals < self.budget:
            # Select the fittest individuals
            self.population = sorted(self.population, key=self.func_evals, reverse=True)[:self.budget]
            
            # Select two parents using tournament selection
            parent1, parent2 = random.sample(self.population, 2)
            
            # Select the best individual to reproduce
            child = self.tournament_selection(parent1, parent2)
            
            # Mutate the child
            child = self.mutate(child)
            
            # Replace the least fit individual with the new child
            self.population[self.func_evals] = child
            
            # Update the best individual and fitness
            self.best_individual = child
            self.best_fitness = self.func_evals
            
            # Update the population size
            self.func_evals += 1
        
        # Return the best individual found
        return self.best_individual

    def tournament_selection(self, individual1, individual2):
        # Select the best individual using tournament selection
        if random.random() < 0.25:
            return individual1
        else:
            return individual2

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Optimizes a black box function using a population-based approach
# with a probability of 0.25 to refine the strategy