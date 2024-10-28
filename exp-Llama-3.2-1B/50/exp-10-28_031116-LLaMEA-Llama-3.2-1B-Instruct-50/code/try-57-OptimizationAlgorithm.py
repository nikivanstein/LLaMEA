import numpy as np
import random

class OptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(self.budget):
            solution = random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)
        
        # Evaluate the fitness of each solution
        fitnesses = [func(solution) for solution in self.population]
        
        # Refine the population using the Pareto Front concept
        for _ in range(self.budget):
            # Select the fittest solutions
            fittest_solutions = self.population[np.argsort(fitnesses)]
            
            # Select a new solution using the Pareto Front concept
            new_solution = fittest_solutions[0]
            while True:
                # Calculate the new fitness
                new_fitness = func(new_solution)
                
                # Check if the new solution is in the Pareto Front
                if new_fitness >= new_solution.fitness:
                    break
                
                # Refine the new solution
                new_solution = self.pareto_refine(new_solution, new_solution.fitness)
        
        # Evaluate the fitness of each solution in the refined population
        self.fitness_scores = [func(solution) for solution in self.population]
        
        # Select the fittest solutions
        self.population = sorted(self.population, key=lambda solution: self.fitness_scores[-1], reverse=True)
        
        # Update the population size
        self.budget -= 1
        
        # If the population size is reduced to zero, stop the algorithm
        if self.budget <= 0:
            break
    
    def pareto_refine(self, solution, fitness):
        # Initialize the new solution
        new_solution = solution
        
        # Refine the new solution using the Pareto Front concept
        while new_solution.fitness < fitness:
            # Select a new solution using the Pareto Front concept
            new_solution = self.pareto_front(new_solution, fitness)
        
        return new_solution
    
    def pareto_front(self, solution, fitness):
        # Initialize the new solution
        new_solution = solution
        
        # Refine the new solution using the Pareto Front concept
        while new_solution.fitness < fitness:
            # Select a new solution using the Pareto Front concept
            new_solution = self.pareto_front(new_solution, fitness)
        
        return new_solution
    
    def select_solution(self):
        # Select the fittest solution
        return self.population[0]