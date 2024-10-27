import numpy as np
import random
import copy
from scipy.optimize import minimize

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

class GeneticProgrammingBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population_size_mutation = 20
        self.population_size_crossover = 5
        self.population_size_evolution = 10
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.evolution_rate = 0.01

    def __call__(self, func):
        # Create a population of random solutions
        population = [copy.deepcopy(self.search(func)) for _ in range(self.population_size)]
        
        # Evaluate the population
        population_evals = 0
        for individual in population:
            func_eval = self.__call__(func, individual)
            population_evals += 1
            if func_eval < self.__call__(func, individual):
                population.remove(individual)
                population.append(individual)
        
        # Select the fittest individuals
        fittest = sorted(population, key=self.__call__, reverse=True)[:self.population_size_mutation]
        
        # Perform mutation
        mutated = []
        for individual in fittest:
            if random.random() < self.mutation_rate:
                mutated.append(individual)
            else:
                mutated.append(self.search(func)(individual))
        
        # Perform crossover
        for i in range(self.population_size_crossover):
            parent1, parent2 = random.sample(fittest, 2)
            child = (parent1 + parent2) / 2
            mutated.append(child)
        
        # Perform evolution
        for _ in range(self.population_size_evolution):
            mutated = sorted(mutated, key=self.__call__, reverse=True)
            mutated = [copy.deepcopy(self.search(func)) for individual in mutated]
            mutated = [individual for individual in mutated if self.__call__(func, individual) >= self.__call__(func, self.search(func))]
        
        # Return the best solution found
        return mutated[0]

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 