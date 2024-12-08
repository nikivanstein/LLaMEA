import numpy as np
import random
from collections import deque
from operator import itemgetter

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
        # Randomly swap two elements in the individual
        i, j = random.sample(range(self.dim), 2)
        individual[i], individual[j] = individual[j], individual[i]
        
        # Ensure the individual is within the bounds
        individual[i] = np.clip(individual[i], -5.0, 5.0)
        individual[j] = np.clip(individual[j], -5.0, 5.0)
        
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select the fittest individual from the parents
        fittest_parent1 = self.search(parent1)
        fittest_parent2 = self.search(parent2)
        
        # Select a random index to crossover
        crossover_index = random.randint(0, self.dim - 1)
        
        # Create a child individual by combining the parents
        child = np.concatenate((parent1[:crossover_index], parent2[crossover_index:]))
        
        # Return the child individual
        return child

    def evolve(self, population_size, mutation_rate, crossover_rate):
        # Initialize the population
        population = [self.search(func) for func in range(self.dim * 100)]
        
        # Evolve the population
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = [individual for individual in population if individual not in [self.search(func) for func in range(self.dim * 100)]]
            
            # Mutate the fittest individuals
            mutated_individuals = [self.mutate(individual) for individual in fittest_individuals]
            
            # Crossover the mutated individuals
            children = [self.crossover(parent1, parent2) for parent1, parent2 in zip(mutated_individuals, mutated_individuals)]
            
            # Replace the old population with the new one
            population = [individual for individual in population if individual not in children]
            population += children
        
        # Return the fittest individual
        return self.search(population[0])

# Example usage:
budget = 1000
dim = 10
metaheuristic = BBOBMetaheuristic(budget, dim)
best_solution = metaheuristic.evolve(population_size=100, mutation_rate=0.1, crossover_rate=0.5)
print("Best solution:", best_solution)
print("Best score:", metaheuristic.search(best_solution))