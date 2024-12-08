import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = deque(maxlen=self.population_size)
        self.population_history = deque(maxlen=100)

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
        # Select a random mutation point
        mutation_point = random.randint(0, self.dim - 1)
        
        # Flip the bit at the mutation point
        individual[mutation_point] = 1 - individual[mutation_point]
        
        # Check if the mutation point is within the bounds
        if not (0 <= individual[mutation_point] <= 1):
            raise ValueError("Mutation point out of bounds")
        
        # Update the population
        self.population.append(individual)

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim - 1)
        
        # Split the parents
        parent1_child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        parent2_child = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        
        # Return the children
        return parent1_child, parent2_child

    def evolve(self):
        # Initialize the population
        self.population.clear()
        
        # Add the initial solution
        self.population.append(self.search(np.random.uniform(-5.0, 5.0, self.dim)))
        
        # Evolve the population
        for _ in range(100):
            # Select the parents
            parent1, parent2 = random.sample(self.population, 2)
            
            # Crossover the parents
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate the children
            self.mutate(child1)
            self.mutate(child2)
            
            # Add the children to the population
            self.population.append(child1)
            self.population.append(child2)
        
        # Return the best solution found
        return self.population[0]

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 