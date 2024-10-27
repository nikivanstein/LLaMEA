import numpy as np
from collections import deque

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = deque(maxlen=1000)
        self.population_size = 100

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
        # Randomly select an individual to mutate
        mutated_individual = individual.copy()
        
        # Randomly swap two random elements in the individual
        idx1 = np.random.randint(0, self.dim)
        idx2 = np.random.randint(0, self.dim)
        
        # Swap the elements at the selected indices
        mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
        
        # Return the mutated individual
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Randomly select a crossover point
        idx = np.random.randint(0, self.dim)
        
        # Create a new child by combining the parents
        child = parent1[:idx] + parent2[idx:]
        
        # Return the child
        return child

    def evaluate_fitness(self, individual, logger):
        # Evaluate the fitness of the individual
        fitness = np.array([self.func_evals - self.func_evals % self.budget])
        
        # Update the logger
        logger.update_fitness(individual, fitness)
        
        # Return the fitness
        return fitness

    def __next__(self):
        # Check if the population is exhausted
        if len(self.population) == 0:
            # Initialize a new population
            new_population = deque(maxlen=self.population_size)
            
            # Generate new individuals using the mutation and crossover operators
            for _ in range(self.population_size):
                parent1 = np.random.uniform(bounds, size=self.dim)
                parent2 = np.random.uniform(bounds, size=self.dim)
                
                # Randomly select a crossover point
                idx = np.random.randint(0, self.dim)
                
                # Create a new child by combining the parents
                child = self.crossover(parent1, parent2)
                
                # Randomly mutate the child
                mutated_child = self.mutate(child)
                
                # Add the mutated child to the new population
                new_population.append(mutated_child)
            
            # Replace the old population with the new population
            self.population = new_population
        
        # Return the next individual
        return self.population.popleft()

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 