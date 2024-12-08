import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def adaptive_line_search(self, func, sol, alpha, beta):
        # Calculate the objective function value
        obj_func_value = func(sol)
        
        # Perform a line search
        step_size = alpha * (obj_func_value - self.func_evals)
        step_direction = np.array([-step_size / np.sqrt(obj_func_value - self.func_evals), step_size / np.sqrt(obj_func_value - self.func_evals)])
        
        # Update the solution using the line search direction
        new_sol = sol + step_direction
        
        # Evaluate the function at the new solution
        func_new_sol = self.__call__(func, new_sol)
        
        # Check if the new solution is better than the current best
        if func_new_sol < self.__call__(func, new_sol):
            # Update the solution
            new_sol = new_sol
        
        # Return the new solution
        return new_sol

    def genetic_algorithm(self, func, bounds, population_size, mutation_rate, alpha, beta):
        # Initialize the population
        population = []
        
        # Create an initial population of random solutions
        for _ in range(population_size):
            individual = np.random.uniform(bounds, size=self.dim)
            population.append(individual)
        
        # Evolve the population using the genetic algorithm
        while len(population) < 100:
            # Evaluate the fitness of each individual in the population
            fitness_values = [self.__call__(func, individual) for individual in population]
            
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness_values)]
            
            # Create a new population by mutating the fittest individuals
            new_population = []
            for _ in range(population_size):
                # Select a random individual from the fittest individuals
                individual = fittest_individuals[np.random.randint(0, len(fittest_individuals))]
                
                # Perform a mutation
                if random.random() < mutation_rate:
                    # Randomly change the solution
                    individual = random.uniform(bounds, size=self.dim)
                    
                    # Evaluate the function at the new solution
                    func_sol = self.__call__(func, individual)
                    
                    # Check if the new solution is better than the current best
                    if func_sol < self.__call__(func, individual):
                        # Update the solution
                        individual = individual
            
            # Add the new individuals to the new population
            new_population.extend([individual] * population_size)
            
            # Update the population
            population = new_population
        
        # Return the fittest individual in the final population
        return population[np.argmin(fitness_values)]