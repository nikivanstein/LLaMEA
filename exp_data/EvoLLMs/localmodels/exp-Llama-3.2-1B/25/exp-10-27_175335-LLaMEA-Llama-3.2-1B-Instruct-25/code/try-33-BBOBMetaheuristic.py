import numpy as np
from collections import deque
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

def mutation(individual, bounds):
    # Randomly mutate the individual
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if random.random() < 0.25:
            mutated_individual[i] += random.uniform(-1, 1)
    return mutated_individual

def selection(population, bounds):
    # Select the fittest individuals
    fitnesses = [self.func_evals for _ in range(len(population))]
    fitnesses.sort(reverse=True)
    selected_indices = [i for i, fitness in enumerate(fitnesses) if fitness == self.func_evals]
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    # Perform crossover between two parents
    child = parent1.copy()
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent2[i]
    return child

def evolution(budget, dim, population_size, mutation_rate, selection_rate):
    # Initialize the population
    population = [random.uniform(bounds, size=dim) for _ in range(population_size)]
    
    # Initialize the best solution
    best_solution = None
    best_fitness = -1
    
    # Run the evolution
    for _ in range(budget):
        # Select the fittest individuals
        population = selection(population, bounds)
        
        # Create a new population
        new_population = []
        for _ in range(population_size):
            # Select two parents
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Perform crossover and mutation
            child = crossover(parent1, parent2)
            child = mutation(child, bounds)
            
            # Add the child to the new population
            new_population.append(child)
        
        # Replace the old population with the new one
        population = new_population
        
        # Check if the best solution has been found
        if best_fitness == -1 or self.func_evals(population) > best_fitness:
            # Update the best solution
            best_solution = population[0]
            best_fitness = self.func_evals(population)
    
    # Return the best solution found
    return best_solution

def func_evals(func, individual):
    # Evaluate the function at the individual
    return self.__call__(func, individual)

# Example usage
budget = 100
dim = 10
population_size = 100
mutation_rate = 0.01
selection_rate = 0.5

best_solution = evolution(budget, dim, population_size, mutation_rate, selection_rate)

# Save the best solution
np.save("currentexp/aucs-BBOBMetaheuristic-" + str(budget) + "-" + str(dim) + "-" + str(population_size) + "-" + str(mutation_rate) + "-" + str(selection_rate) + ".npy", best_solution)