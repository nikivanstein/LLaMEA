# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Adaptive Mutation
# Code: 
# ```python
import numpy as np
import random
from collections import deque

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.mutation_rate = 0.1
        self.search_space_bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        self.population = deque(maxlen=self.budget)

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
        bounds = np.linspace(self.search_space_bounds[0], self.search_space_bounds[1], self.dim, endpoint=False)
        
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
        # Generate a new mutation
        mutation = random.uniform(self.search_space_bounds[0], self.search_space_bounds[1])
        
        # Update the individual
        new_individual = individual.copy()
        new_individual[self.mutation_index] = mutation
        
        # Check if the mutation is within the bounds
        if self.mutation_index < len(individual) and individual[self.mutation_index] > self.search_space_bounds[0]:
            # Update the mutation index
            self.mutation_index += 1
            if self.mutation_index >= len(individual):
                self.mutation_index = 0
        
        # Check if the mutation is within the bounds of the budget
        if self.func_evals + 1 >= self.budget:
            # Update the population
            self.population.append(individual)
        
        # Return the new individual
        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        fitness = self.func(individual)
        
        # Update the individual
        individual = self.mutate(individual)
        
        # Check if the individual is within the bounds
        if individual[0] < -5.0 or individual[0] > 5.0:
            # Update the bounds
            individual[0] = -5.0
        if individual[-1] < -5.0 or individual[-1] > 5.0:
            # Update the bounds
            individual[-1] = 5.0
        
        # Return the fitness
        return fitness

    def next_generation(self):
        # Initialize the next generation
        next_generation = deque(maxlen=self.budget)
        
        # Iterate over the population
        for individual in self.population:
            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual)
            
            # Add the individual to the next generation
            next_generation.append(individual)
        
        # Return the next generation
        return next_generation

    def run(self, func):
        # Run the algorithm for a specified number of generations
        for _ in range(100):
            # Get the next generation
            next_generation = self.next_generation()
            
            # Evaluate the fitness of the next generation
            fitnesses = [self.evaluate_fitness(individual) for individual in next_generation]
            
            # Get the best individual
            best_individual = min(next_generation, key=fitnesses)
            
            # Update the best individual
            self.best_individual = best_individual
            
            # Print the best individual
            print("Best Individual:", best_individual)
        
        # Return the best individual
        return self.best_individual

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Adaptive Mutation
# Code: 
# ```python
# import numpy as np
# import random
# from collections import deque

bboom = BBOBMetaheuristic(1000, 10)
bboom.run(lambda func: np.sin(np.linspace(-5.0, 5.0, 1000)))