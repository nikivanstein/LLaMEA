import numpy as np
import random
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

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

    def search(self, func, bounds, mutation_prob=0.25, mutation_size=1.0):
        # Define the search space
        bounds = np.linspace(bounds[0], bounds[1], self.dim, endpoint=False)
        
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
        
        # Refine the solution using genetic programming
        for _ in range(100):
            # Select parents using tournament selection
            parents = self.select_parents(sol, bounds)
            
            # Crossover (reproduce) the parents to create offspring
            offspring = self.crossover(parents)
            
            # Mutate the offspring with a probability based on the mutation size
            offspring = self.mutate(offspring, mutation_prob, mutation_size)
            
            # Replace the current solution with the offspring
            sol = offspring
        
        # Return the best solution found
        return sol

    def select_parents(self, sol, bounds):
        # Select parents using tournament selection
        parents = []
        for _ in range(10):
            # Randomly select two parents
            parent1 = np.random.uniform(bounds[0], bounds[1], self.dim)
            parent2 = np.random.uniform(bounds[0], bounds[1], self.dim)
            
            # Evaluate the function at both parents
            func_sol1 = self.__call__(func, parent1)
            func_sol2 = self.__call__(func, parent2)
            
            # Choose the parent with the better solution
            if func_sol1 < func_sol2:
                parents.append(parent1)
            else:
                parents.append(parent2)
        
        # Return the selected parents
        return parents

    def crossover(self, parents):
        # Crossover (reproduce) the parents to create offspring
        offspring = []
        for _ in range(len(parents)):
            # Choose a random crossover point
            crossover_point = random.randint(0, len(parents) - 1)
            
            # Create the offspring
            offspring.append(parents[crossover_point])
            
            # Crossover the offspring
            for i in range(len(parents[crossover_point]) - 1):
                # Choose a random crossover point
                crossover_point1 = random.randint(0, i)
                crossover_point2 = random.randint(crossover_point1 + 1, len(parents[crossover_point]) - 1)
                
                # Create the offspring
                offspring.append(np.concatenate((parents[crossover_point1], parents[crossover_point2])))
        return offspring

    def mutate(self, offspring, mutation_prob, mutation_size):
        # Mutate the offspring with a probability based on the mutation size
        mutated_offspring = []
        for individual in offspring:
            # Randomly decide to mutate or not
            if random.random() < mutation_prob:
                # Mutate the individual
                mutated_individual = individual.copy()
                for i in range(self.dim):
                    # Randomly decide whether to change the individual
                    if random.random() < mutation_size:
                        mutated_individual[i] += random.uniform(-mutation_size, mutation_size)
                mutated_offspring.append(mutated_individual)
            else:
                mutated_offspring.append(individual)
        return mutated_offspring

# Define the function to optimize
def func(x):
    return x[0]**2 + x[1]**2

# Create an instance of the BBOBMetaheuristic class
bboo = BBOBMetaheuristic(100, 2)

# Optimize the function using the evolutionary algorithm
sol = bboo.search(func, bounds=[-5.0, 5.0], mutation_prob=0.5, mutation_size=1.0)

# Print the best solution found
print("Best solution:", sol)