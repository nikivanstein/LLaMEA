import numpy as np
import random
from collections import deque
import copy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = deque(maxlen=1000)

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
        # Randomly change a single line of the individual
        i = random.randint(0, self.dim - 1)
        j = random.randint(0, self.dim - 1)
        while j == i:
            j = random.randint(0, self.dim - 1)
        individual[i], individual[j] = individual[j], individual[i]
        
        # Update the fitness of the individual
        self.population[(individual, self.func(individual))].fitness += 1

    def crossover(self, parent1, parent2):
        # Randomly choose a crossover point
        i = random.randint(0, self.dim - 1)
        j = random.randint(0, self.dim - 1)
        
        # Combine the parents
        child = np.concatenate((parent1[:i], parent2[i:]), axis=0)
        
        # Update the fitness of the child
        self.population[(child, self.func(child))].fitness += 1

    def selection(self, parents):
        # Select the fittest individuals
        self.population = deque(sorted(self.population, key=lambda x: x.fitness, reverse=True), maxlen=1000)

# One-line description with the main idea:
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# This algorithm uses genetic programming to evolve a population of individuals, each representing a possible solution to the black box optimization problem.
# The algorithm starts with a random population and iteratively applies crossover, mutation, and selection to refine the strategy.
# The fitness of each individual is updated based on its performance in the function evaluations.
# The algorithm terminates when a satisfactory solution is found or a maximum number of evaluations is reached.

# Code: