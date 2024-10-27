import numpy as np
import random
from copy import deepcopy
from typing import Dict, List

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = self.initialize_population()

    def __call__(self, func: np.ndarray) -> np.ndarray:
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func: np.ndarray) -> np.ndarray:
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

    def initialize_population(self) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Add the solution to the population
            population.append(deepcopy(sol))
        
        return population

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        # Randomly select a single gene to mutate
        idx = random.randint(0, self.dim - 1)
        
        # Generate a new individual by flipping the mutated gene
        new_individual = individual.copy()
        new_individual[idx] = 1 - new_individual[idx]
        
        # Return the mutated individual
        return new_individual

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Select a random crossover point
        idx = random.randint(0, self.dim - 1)
        
        # Create a new child by combining the two parents
        child = np.concatenate((parent1[:idx], parent2[idx:]), axis=0)
        
        # Return the child
        return child

    def evaluate_fitness(self, individual: np.ndarray) -> float:
        # Evaluate the function at the individual
        func_sol = self.__call__(func, individual)
        
        # Return the fitness value
        return func_sol

    def fitness(self, individual: np.ndarray) -> float:
        # Calculate the fitness value using the fitness function
        fitness = self.evaluate_fitness(individual)
        
        # Update the fitness value
        self.func_evals += 1
        
        return fitness

# One-line description with the main idea:
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# This algorithm uses a population of individuals, each representing a possible solution to the optimization problem.
# The algorithm evaluates the fitness of each individual using the given function and selects the fittest individuals for the next generation.
# The process is repeated until the budget is exhausted.