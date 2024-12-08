import numpy as np
import random
from collections import deque
import time

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.mutation_rate = 0.1
        self.mutation_history = deque(maxlen=100)

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

def mutation(individual, mutation_rate):
    # Randomly mutate the individual
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] += random.uniform(-1, 1)
    return mutated_individual

def fitness(individual, func, budget):
    # Evaluate the function at the individual
    func_eval = self.__call__(func, individual)
    
    # Update the evaluation count
    self.func_evals += 1
    
    # Check if the function has been evaluated enough
    if self.func_evals >= budget:
        raise ValueError("Not enough evaluations left to optimize the function")
    
    # Calculate the fitness score
    fitness = func_eval / budget
    return fitness

def genetic_programming(individual, func, budget, mutation_rate):
    # Initialize the population
    population = [individual]
    
    # Evaluate the function for each individual in the population
    for _ in range(100):
        # Evaluate the function for each individual
        fitnesses = [fitness(individual, func, budget) for individual in population]
        
        # Select the fittest individuals
        fittest_individuals = [individual for individual, fitness in zip(population, fitnesses) if fitness > max(fitnesses)]
        
        # Create a new generation of individuals
        new_population = []
        for _ in range(10):
            # Select two parents using tournament selection
            parent1 = random.choice(fittest_individuals)
            parent2 = random.choice(fittest_individuals)
            
            # Create a new individual by combining the parents
            new_individual = mutation(parent1, mutation_rate)
            
            # Evaluate the new individual
            fitness = fitness(new_individual, func, budget)
            
            # Add the new individual to the new population
            new_population.append(new_individual)
        
        # Replace the old population with the new population
        population = new_population
    
    # Return the best individual in the new population
    return max(population, key=fitness)

# Initialize the genetic programming algorithm
algo = BBOBMetaheuristic(100, 10)
best_individual = None
best_fitness = float('-inf')
best_score = float('-inf')

# Run the genetic programming algorithm
for _ in range(100):
    # Select the fittest individual
    individual = genetic_programming(algo.search, lambda x: x, 100, 0.1)
    
    # Evaluate the function for the individual
    fitness = fitness(individual, lambda x: x, 100)
    
    # Update the best individual and best score
    if fitness > best_fitness:
        best_individual = individual
        best_fitness = fitness
        best_score = fitness
    elif fitness == best_fitness:
        best_individual = mutation(best_individual, 0.1)

# Print the final best individual and its fitness score
print("Best Individual:", best_individual)
print("Best Fitness Score:", best_score)