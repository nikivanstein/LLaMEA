import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

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

    def genetic_optimization(self, func, budget, dim, mutation_rate=0.01):
        # Initialize the population
        population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(budget)]
        
        # Evaluate the function for the first individuals
        for func_sol in population:
            func_sol = self.__call__(func, func_sol)
        
        # Calculate the fitness of each individual
        fitness = np.array([func_sol / 10 for func_sol in population])
        
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[-budget:]
        
        # Create the new population
        new_population = []
        for _ in range(budget):
            # Select two parents using tournament selection
            parent1, parent2 = np.random.choice(fittest_individuals, size=2, replace=False)
            
            # Create the offspring
            offspring = np.random.uniform(-5.0, 5.0, dim)
            
            # Evaluate the function at the offspring
            func_offspring = self.__call__(func, offspring)
            
            # Check if the offspring is better than the current best
            if func_offspring < self.__call__(func, offspring):
                # Update the offspring
                offspring = parent1
            
            # Add the offspring to the new population
            new_population.append(offspring)
        
        # Mutate the new population
        new_population = np.array([self.mutation(individual, mutation_rate) for individual in new_population])
        
        # Replace the old population with the new population
        population = new_population
        
        # Return the best solution found
        return population[np.argmax(fitness)]

    def mutation(self, individual, mutation_rate):
        # Generate a new individual
        new_individual = individual.copy()
        
        # Randomly select a point in the search space
        index = np.random.randint(0, self.dim)
        
        # Update the new individual
        new_individual[index] += np.random.uniform(-1, 1) * 0.1
        
        # Check if the new individual is within the bounds
        if new_individual[index] < -5.0:
            new_individual[index] = -5.0
        elif new_individual[index] > 5.0:
            new_individual[index] = 5.0
        
        # Apply mutation rate
        if np.random.rand() < mutation_rate:
            new_individual[index] += np.random.uniform(-1, 1)
        
        # Return the new individual
        return new_individual

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
# import random
# import matplotlib.pyplot as plt

# class BBOBMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0

#     def __call__(self, func):
#         # Check if the function can be evaluated within the budget
#         if self.func_evals >= self.budget:
#             raise ValueError("Not enough evaluations left to optimize the function")

#         # Evaluate the function within the budget
#         func_evals = self.func_evals
#         self.func_evals += 1
#         return func

#     def search(self, func):
#         # Define the search space
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
#         # Initialize the solution
#         sol = None
        
#         # Try different initializations
#         for _ in range(10):
#             # Randomly initialize the solution
#             sol = np.random.uniform(bounds, size=self.dim)
            
#             # Evaluate the function at the solution
#             func_sol = self.__call__(func, sol)
            
#             # Check if the solution is better than the current best
#             if func_sol < self.__call__(func, sol):
#                 # Update the solution
#                 sol = sol
        
#         # Return the best solution found
#         return sol

# def genetic_optimization(func, budget, dim, mutation_rate=0.01):
#     # Initialize the population
#     population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(budget)]
        
#     # Evaluate the function for the first individuals
#     for func_sol in population:
#         func_sol = func_sol / 10  # Normalize the function
    
#     # Calculate the fitness of each individual
#     fitness = np.array([func_sol / 10 for func_sol in population])
    
#     # Select the fittest individuals
#     fittest_individuals = np.argsort(fitness)[-budget:]
    
#     # Create the new population
#     new_population = []
#     for _ in range(budget):
#         # Select two parents using tournament selection
#         parent1, parent2 = np.random.choice(fittest_individuals, size=2, replace=False)
            
#         # Create the offspring
#         offspring = np.random.uniform(-5.0, 5.0, dim)
            
#         # Evaluate the function at the offspring
#         func_offspring = func.offspring
#         func_offspring = func_offspring / 10  # Normalize the function
            
#         # Check if the offspring is better than the current best
#         if func_offspring < self.__call__(func, offspring):
#             # Update the offspring
#             offspring = parent1
            
#         # Add the offspring to the new population
#         new_population.append(offspring)
        
#     # Mutate the new population
#     new_population = np.array([self.mutation(individual, mutation_rate) for individual in new_population])
        
#     # Replace the old population with the new population
#     population = new_population
        
#     # Return the best solution found
#     return population[np.argmax(fitness)]

# def mutation(individual, mutation_rate):
#     # Generate a new individual
#     new_individual = individual.copy()
        
#     # Randomly select a point in the search space
#     index = np.random.randint(0, self.dim)
        
#     # Update the new individual
#     new_individual[index] += np.random.uniform(-1, 1) * 0.1
        
#     # Check if the new individual is within the bounds
#     if new_individual[index] < -5.0:
#         new_individual[index] = -5.0
#     elif new_individual[index] > 5.0:
#         new_individual[index] = 5.0
        
#     # Apply mutation rate
#     if np.random.rand() < mutation_rate:
#         new_individual[index] += np.random.uniform(-1, 1)
        
#     # Return the new individual
#     return new_individual

# def plot_fitness(func, population, mutation_rate):
#     # Plot the fitness of each individual
#     plt.plot(population, func(population))
#     plt.xlabel('Fitness')
#     plt.ylabel('Individual')
#     plt.title('Fitness of Each Individual')
#     plt.show()

# # Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# # Code: 
# # ```python
# # import numpy as np
# # import scipy.optimize as optimize
# # import random
# # import matplotlib.pyplot as plt

# # class BBOBMetaheuristic:
# #     def __init__(self, budget, dim):
# #         self.budget = budget
# #         self.dim = dim
# #         self.func_evals = 0

# #     def __call__(self, func):
# #         # Check if the function can be evaluated within the budget
# #         if self.func_evals >= self.budget:
# #             raise ValueError("Not enough evaluations left to optimize the function")

# #         # Evaluate the function within the budget
# #         func_evals = self.func_evals
# #         self.func_evals += 1
# #         return func

# #     def search(self, func):
# #         # Define the search space
# #         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
# #         # Initialize the solution
# #         sol = None
        
# #         # Try different initializations
# #         for _ in range(10):
# #             # Randomly initialize the solution
# #             sol = np.random.uniform(bounds, size=self.dim)
            
# #             # Evaluate the function at the solution
# #             func_sol = self.__call__(func, sol)
            
# #             # Check if the solution is better than the current best
# #             if func_sol < self.__call__(func, sol):
# #                 # Update the solution
# #                 sol = sol
        
# #         # Return the best solution found
# #         return sol

# def genetic_optimization(func, budget, dim, mutation_rate=0.01):
#     # Initialize the population
#     population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(budget)]
        
#     # Evaluate the function for the first individuals
#     for func_sol in population:
#         func_sol = func_sol / 10  # Normalize the function
    
#     # Calculate the fitness of each individual
#     fitness = np.array([func_sol / 10 for func_sol in population])
    
#     # Select the fittest individuals
#     fittest_individuals = np.argsort(fitness)[-budget:]
    
#     # Create the new population
#     new_population = []
#     for _ in range(budget):
#         # Select two parents using tournament selection
#         parent1, parent2 = np.random.choice(fittest_individuals, size=2, replace=False)
            
#         # Create the offspring
#         offspring = np.random.uniform(-5.0, 5.0, dim)
            
#         # Evaluate the function at the offspring
#         func_offspring = func.offspring
#         func_offspring = func_offspring / 10  # Normalize the function
            
#         # Check if the offspring is better than the current best
#         if func_offspring < self.__call__(func, offspring):
#             # Update the offspring
#             offspring = parent1
            
#         # Add the offspring to the new population
#         new_population.append(offspring)
        
#     # Mutate the new population
#     new_population = np.array([self.mutation(individual, mutation_rate) for individual in new_population])
        
#     # Replace the old population with the new population
#     population = new_population
        
#     # Return the best solution found
#     return population[np.argmax(fitness)]

# def plot_fitness(func, population, mutation_rate):
#     # Plot the fitness of each individual
#     plt.plot(population, func(population))
#     plt.xlabel('Fitness')
#     plt.ylabel('Individual')
#     plt.title('Fitness of Each Individual')
#     plt.show()

# # main function
# def main():
#     # Define the function to optimize
#     func = lambda x: x**2
    
#     # Define the budget and dimension
#     budget = 100
#     dim = 10
    
#     # Create the genetic optimization algorithm
#     algorithm = BBOBMetaheuristic(budget, dim)
    
#     # Run the algorithm
#     best_solution = algorithm.search(func, budget, dim)
    
#     # Print the result
#     print(f'Best solution: {best_solution}')
    
#     # Plot the fitness of each individual
#     plot_fitness(func, algorithm.population, 0.1)

# # Call the main function
# main()