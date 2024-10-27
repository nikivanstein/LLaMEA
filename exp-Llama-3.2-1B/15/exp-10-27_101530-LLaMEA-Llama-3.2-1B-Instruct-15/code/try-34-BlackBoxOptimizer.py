import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def novel_metaheuristic(self, func, budget, dim):
        # Define the mutation function
        def mutate(individual):
            # Generate a new individual by refining the current one
            new_individual = individual + (random.uniform(-1, 1), random.uniform(-1, 1))
            # Check if the new individual is within the budget
            if self.func_evaluations < budget:
                # If not, return the new individual
                return new_individual
            # If the new individual is within the budget, return the current individual
            return individual
        
        # Define the selection function
        def select(population, budget):
            # Select the fittest individuals
            fittest = population[np.argsort(self.func_evaluations)][-budget:]
            # Select individuals with a fitness score below 0.15
            return fittest[:int(budget * 0.15)]

        # Define the crossover function
        def crossover(parent1, parent2):
            # Generate a new individual by combining the parents
            child = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
            # Check if the child is within the budget
            if self.func_evaluations < budget:
                # If not, return the child
                return child
            # If the child is within the budget, return the current parents
            return parent1, parent2

        # Initialize the population
        population = [self.novel_metaheuristic(func, budget, dim) for _ in range(100)]

        # Run the selection, crossover, and mutation processes
        while len(population) > 0:
            # Select the fittest individuals
            population = select(population, budget)
            # Generate new individuals
            new_population = []
            for _ in range(100):
                parent1, parent2 = random.sample(population, 2)
                child = self.novel_metaheuristic(func, budget, dim)
                while child == parent1 or child == parent2:
                    child = self.novel_metaheuristic(func, budget, dim)
                new_population.append(child)
            population = new_population

        # Return the best individual found
        return population[0]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 